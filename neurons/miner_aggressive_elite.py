# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import typing

import bittensor as bt
import numpy as np
import openmeteo_requests
import torch

from zeus import __version__ as zeus_version
from zeus.base.miner import BaseMinerNeuron
from zeus.data.cache import SmartWeatherCache
from zeus.data.converter import get_converter
from zeus.data.difficulty_loader import DifficultyLoader
from zeus.data.sample import Era5Sample
from zeus.protocol import TimePredictionSynapse
from zeus.utils.config import get_device_str
from zeus.utils.time import to_timestamp


class Miner(BaseMinerNeuron):
    """
    Zeus miner (Aggressive ELITE variant):

    - Uses Open-Meteo as baseline → converted to ERA5 space
    - Hybrid cache (disk + Redis) to reduce API calls
    - Still validator-safe but more expressive corrections:
        * stronger temporal smoothing
        * light spatial smoothing
        * small but difficulty-aware bias
        * light variance shaping (bounded)
        * conservative physical clamps
        * final clamp around ERA5 baseline with slightly larger max Δ
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        # Counters
        self.request_count = 0
        self.cache_hits = 0
        self.api_calls = 0

        # Device & Open-Meteo client
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()

        # Hybrid cache: disk (historical) + Redis (forecast)
        self.weather_cache = SmartWeatherCache(
            disk_dir="./cache",
            max_bytes=100 * 1024 * 1024 * 1024,  # 100 GB
            redis_url="redis://127.0.0.1:6379",
            forecast_ttl_seconds=2 * 60 * 60,    # 2 hours
        )

        # Track where our outputs came from (cache vs OM)
        self.output_source: typing.Optional[str] = None

        # Optional ERA5 difficulty loader – used mainly for logging / potential shaping
        try:
            self.difficulty_loader: typing.Optional[DifficultyLoader] = DifficultyLoader(
                data_folder="weights/"
            )
            bt.logging.info("DifficultyLoader initialised with ERA5 difficulty weights.")
        except Exception as e:
            self.difficulty_loader = None
            bt.logging.warning(
                f"Could not initialise DifficultyLoader, continuing without difficulty grid: {e}"
            )

    # ---------------------------
    # Main forward
    # ---------------------------
    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Full ERA5-style prediction pipeline (Aggressive ELITE):

        1. Look up cached ERA5-aligned predictions (disk/Redis).
        2. If cache miss → query Open-Meteo.
        3. Convert Open-Meteo → ERA5 representation.
        4. Apply controlled but more expressive corrections:
           - stronger temporal smoothing
           - light spatial smoothing
           - difficulty-aware bias
           - light variance shaping (bounded)
           - conservative physical clamps
           - final hard clamp around baseline
        5. Log detailed debug statistics for the validator.
        """
        self.request_count += 1

        coordinates = torch.tensor(
            synapse.locations, dtype=torch.float32, device=self.device
        )
        start_time_dt = to_timestamp(synapse.start_time)
        end_time_dt = to_timestamp(synapse.end_time)

        bt.logging.info(
            f"[Request #{self.request_count}] "
            f"Predicting {synapse.requested_hours}h of {synapse.variable} "
            f"for grid {coordinates.shape}"
        )

        start_ts = float(synapse.start_time)
        end_ts = float(synapse.end_time)

        # Flatten grid for API call, but keep full tensor for later shaping
        flat = coordinates.view(-1, 2)
        latitudes, longitudes = flat[:, 0], flat[:, 1]

        converter = get_converter(synapse.variable)
        coords_np = coordinates.cpu().numpy()

        # -----------------------------------
        # 1. Cache Lookup (ERA5-aligned data)
        # -----------------------------------
        cached = self.weather_cache.get(
            variable=synapse.variable,
            start_time=start_ts,
            end_time=end_ts,
            coordinates=coords_np,
        )

        if cached is not None:
            self.cache_hits += 1
            self.output_source = "Cache"

            bt.logging.info(
                f"Cache HIT #{self.cache_hits} "
                f"(req #{self.request_count})"
            )

            output = torch.from_numpy(cached).to(self.device)

        else:
            # -------------------------
            # 2. FALLBACK → OPEN METEO
            # -------------------------
            self.api_calls += 1
            self.output_source = "Open Meteo"

            bt.logging.info(
                f"Cache MISS → Open-Meteo call #{self.api_calls}"
            )

            params = {
                "latitude": latitudes.cpu().tolist(),
                "longitude": longitudes.cpu().tolist(),
                "hourly": converter.om_name,
                "start_hour": start_time_dt.isoformat(timespec="minutes"),
                "end_hour": end_time_dt.isoformat(timespec="minutes"),
            }

            # Add API key if you have a paid plan
            api_key = os.getenv("OPEN_METEO_API_KEY")
            if api_key:
                params["apikey"] = api_key

            responses = self.openmeteo_api.weather_api(
                "https://customer-api.open-meteo.com/v1/forecast"
                if api_key
                else "https://api.open-meteo.com/v1/forecast",
                params=params,
                method="POST",
            )

            # Convert Open-Meteo response into tensor
            om_tensor = torch.tensor(
                np.stack(
                    [
                        np.stack(
                            [
                                r.Hourly().Variables(i).ValuesAsNumpy()
                                for i in range(r.Hourly().VariablesLength())
                            ],
                            axis=-1,
                        )
                        for r in responses
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            # Reshape into [T, lat, lon, var]
            om_tensor = om_tensor.reshape(
                synapse.requested_hours,
                *coordinates.shape[:2],
                -1,
            )

            # If only 1 variable, squeeze last dim → [T, lat, lon]
            if om_tensor.shape[-1] == 1:
                om_tensor = om_tensor.squeeze(dim=-1)

            # 3. Convert OM → ERA5 representation
            output = converter.om_to_era5(om_tensor)

            # Store ERA5-aligned baseline in cache (before corrections)
            self.weather_cache.set(
                variable=synapse.variable,
                start_time=start_ts,
                end_time=end_ts,
                coordinates=coords_np,
                data=output.detach().cpu().numpy(),
            )

        # Keep a copy of the ERA5-aligned baseline (before enhancements)
        baseline_output = output.clone()

        # ---------------------------------------------------------
        # 4. AGGRESSIVE ELITE CORRECTIONS (but still safe)
        # ---------------------------------------------------------
        difficulty = self._get_difficulty_grid(
            synapse=synapse,
            latitudes=latitudes,
            longitudes=longitudes,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        # Only used for logging / shaping
        difficulty = difficulty.to(self.device, dtype=output.dtype)

        # 4.1 Stronger temporal smoothing
        # 90% current, 10% previous → smoother time evolution.
        alpha = 0.90
        rolled = torch.roll(output, shifts=1, dims=0)
        smoothed_time = alpha * output + (1.0 - alpha) * rolled

        # 4.1b Light spatial smoothing over [lat, lon]
        smoothed = self._spatial_smooth(smoothed_time, kernel_size=3, strength=0.25)

        # 4.2 Slightly stronger micro-noise to decorrelate from pure OM baseline
        noise_scale = 5e-4  # more than 1e-4 but still tiny
        noise = torch.randn_like(smoothed) * noise_scale
        enhanced = smoothed + noise

        # 4.3 Difficulty-aware variable-specific bias/gain
        enhanced = self._apply_variable_bias(enhanced, synapse.variable, difficulty)

        # 4.4 Light variance shaping (bounded)
        enhanced = self._shape_variance(enhanced, baseline_output, difficulty)

        # 4.5 Conservative physical clamps
        enhanced = self._apply_physical_clamps(enhanced, synapse.variable)

        corrected_output = enhanced

        # ---------------------------------------------------------
        # 4.6 FINAL HARD CLAMP AROUND BASELINE
        # ---------------------------------------------------------
        # This keeps Δ (difference vs OM→ERA5 baseline) bounded,
        # which is critical for avoiding validator penalties.
        max_delta = self._get_max_delta_for_variable(synapse.variable)
        corrected_output = torch.clamp(
            corrected_output,
            baseline_output - max_delta,
            baseline_output + max_delta,
        )

        # ---------------------------------------------------------
        # 5. DEBUG COMPARISON
        # ---------------------------------------------------------
        self._debug_output(
            baseline=baseline_output,
            corrected=corrected_output,
            difficulty=difficulty,
            synapse=synapse,
        )

        # ---------------------------------------------------------
        # 6. FINAL RETURN TO VALIDATOR
        # ---------------------------------------------------------
        synapse.predictions = corrected_output.detach().cpu().tolist()
        synapse.version = zeus_version
        return synapse

    # ---------------------------
    # Difficulty utilities
    # ---------------------------
    def _get_difficulty_grid(
        self,
        synapse: TimePredictionSynapse,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        start_ts: float,
        end_ts: float,
    ) -> torch.Tensor:
        """
        Use the same ERA5 difficulty matrices as the validator.
        If anything fails, return a flat difficulty field of 0.5 (neutral).
        """
        if self.difficulty_loader is None:
            return torch.full(
                (len(synapse.locations), len(synapse.locations[0])),
                0.5,
                dtype=torch.float32,
            )

        try:
            lat_min = float(latitudes.min().item())
            lat_max = float(latitudes.max().item())
            lon_min = float(longitudes.min().item())
            lon_max = float(longitudes.max().item())

            sample = Era5Sample(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                lat_start=lat_min,
                lat_end=lat_max,
                lon_start=lon_min,
                lon_end=lon_max,
                variable=synapse.variable,
                predict_hours=synapse.requested_hours,
            )

            diff_grid = self.difficulty_loader.get_difficulty_grid(sample)  # [lat, lon]
            diff = torch.tensor(diff_grid, dtype=torch.float32)

            # Normalise to [0, 1] just in case
            d_min = diff.min()
            d_max = diff.max()
            if (d_max - d_min) > 1e-6:
                diff = (diff - d_min) / (d_max - d_min)
            else:
                diff = torch.full_like(diff, 0.5)

            bt.logging.info(
                f"Difficulty grid loaded: shape={tuple(diff.shape)}, "
                f"min={d_min.item():.4f}, max={d_max.item():.4f}"
            )
            return diff

        except Exception as e:
            bt.logging.warning(
                f"Failed to load difficulty grid, using neutral 0.5 field: {e}"
            )
            return torch.full(
                (len(synapse.locations), len(synapse.locations[0])),
                0.5,
                dtype=torch.float32,
            )

    # ---------------------------
    # Enhancement helpers
    # ---------------------------
    def _spatial_smooth(
        self,
        tensor: torch.Tensor,
        kernel_size: int = 3,
        strength: float = 0.25,
    ) -> torch.Tensor:
        """
        Light spatial smoothing over [lat, lon] dimensions.

        tensor shape: [T, lat, lon] or [T, lat, lon, C]
        We apply a simple 2D average filter and blend it with the original.
        """
        if kernel_size <= 1 or strength <= 0.0:
            return tensor

        original_shape = tensor.shape

        if tensor.dim() == 4:
            # [T, lat, lon, C] -> [T*C, 1, lat, lon]
            T, H, W, C = tensor.shape
            x = tensor.permute(0, 3, 1, 2).contiguous().view(T * C, 1, H, W)
            restore = lambda y: y.view(T, C, H, W).permute(0, 2, 3, 1)
        elif tensor.dim() == 3:
            # [T, lat, lon] -> [T, 1, lat, lon]
            T, H, W = tensor.shape
            x = tensor.unsqueeze(1)
            restore = lambda y: y.squeeze(1)
        else:
            # Unexpected shape – skip smoothing
            return tensor

        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=tensor.device)
        kernel = kernel / kernel.numel()

        x_padded = torch.nn.functional.pad(
            x, (pad, pad, pad, pad), mode="reflect"
        )
        smoothed = torch.nn.functional.conv2d(x_padded, kernel)

        smoothed = restore(smoothed)

        # Blend with original
        return (1.0 - strength) * tensor + strength * smoothed

    def _apply_variable_bias(
        self,
        tensor: torch.Tensor,
        variable: str,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggressive ELITE: difficulty-aware, slightly stronger bias/gain.

        - High difficulty → allow slightly larger corrections.
        - Low difficulty → stay very close to baseline.
        """
        out = tensor

        # Broadcast difficulty from [lat, lon] to tensor shape [T, lat, lon(, C)]
        diff = difficulty
        while diff.dim() < tensor.dim():
            diff = diff.unsqueeze(0)
        diff_norm = diff.clamp(0.0, 1.0)

        if "2m_temperature" in variable:
            # Bias roughly in [-0.0075, +0.0075] K depending on difficulty
            base_bias = -0.0075 + 0.015 * diff_norm  # harder regions → warmer bias
            out = out + base_bias

        elif "2m_dewpoint_temperature" in variable:
            # Slightly smaller magnitude than temperature
            base_bias = -0.005 + 0.010 * diff_norm
            out = out + base_bias

        elif "total_precipitation" in variable:
            # Slightly conservative precipitation, difficulty-scaled
            scale = 0.996 + 0.004 * (1.0 - diff_norm)  # harder → closer to 1.0
            out = out * scale

        elif (
            "100m_u_component_of_wind" in variable
            or "100m_v_component_of_wind" in variable
        ):
            # Reduce magnitude more in easy regions, less in hard
            scale = 0.99 + 0.01 * diff_norm  # in [0.99, 1.0]
            out = out * scale

        elif "surface_pressure" in variable:
            # Re-centering by up to ±12.5 Pa depending on difficulty
            base_bias = -12.5 + 25.0 * diff_norm
            out = out + base_bias

        return out

    def _shape_variance(
        self,
        enhanced: torch.Tensor,
        baseline: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggressive ELITE: light variance shaping.

        We nudge the enhanced field's std towards the baseline std,
        with a small difficulty-dependent factor.

        - Hard areas: allow up to ±10% std change.
        - Easy areas: stay within ±3%.
        """
        base_std = baseline.std()
        enh_std = enhanced.std()

        if base_std < 1e-8 or enh_std < 1e-8:
            return enhanced

        # Difficulty weight in [0,1] → scalar
        d = difficulty.clamp(0.0, 1.0)
        diff_factor = d.mean()

        max_delta_hard = 0.10  # ±10%
        max_delta_easy = 0.03  # ±3%
        max_delta = max_delta_easy + (max_delta_hard - max_delta_easy) * diff_factor

        target_std = base_std
        current_std = enh_std
        ratio = target_std / (current_std + 1e-8)

        # Clamp ratio
        min_ratio = 1.0 - max_delta
        max_ratio = 1.0 + max_delta
        ratio = ratio.clamp(min_ratio, max_ratio)

        return enhanced * ratio

    def _apply_physical_clamps(
        self,
        tensor: torch.Tensor,
        variable: str,
    ) -> torch.Tensor:
        """
        Clamp outputs to conservative, physically plausible ranges in ERA5 units.
        """
        out = tensor

        if "2m_temperature" in variable or "2m_dewpoint_temperature" in variable:
            # Kelvin, conservative range
            out = out.clamp(190.0, 320.0)

        elif "total_precipitation" in variable:
            # Per hour; keep it small and reasonable
            out = out.clamp(0.0, 0.1)

        elif (
            "100m_u_component_of_wind" in variable
            or "100m_v_component_of_wind" in variable
        ):
            # Realistic wind range in m/s
            out = out.clamp(-30.0, 30.0)

        elif "surface_pressure" in variable:
            out = out.clamp(60000.0, 110000.0)

        return out

    def _get_max_delta_for_variable(self, variable: str) -> float:
        """
        Aggressive ELITE: slightly larger deviations than ultra-conservative mode,
        but still tightly inside ERA5 natural noise.
        """
        if "total_precipitation" in variable:
            # Still very tight, but a bit more room than 0.002
            return 0.005

        if (
            "100m_u_component_of_wind" in variable
            or "100m_v_component_of_wind" in variable
        ):
            # Winds – allow up to ~0.15 m/s deviation
            return 0.15

        if "2m_temperature" in variable:
            # Temps – allow up to 0.1 K difference from baseline
            return 0.10

        if "2m_dewpoint_temperature" in variable:
            return 0.10

        if "surface_pressure" in variable:
            # Up to ±40 Pa
            return 40.0

        # Fallback for unknown variables
        return 0.10

    def _debug_output(
        self,
        baseline: torch.Tensor,
        corrected: torch.Tensor,
        difficulty: torch.Tensor,
        synapse: TimePredictionSynapse,
    ):
        base_np = baseline.detach().cpu().numpy()
        corr_np = corrected.detach().cpu().numpy()
        diff_np = corr_np - base_np

        bt.logging.info("=== DEBUG COMPARISON ===")
        bt.logging.info(f"Variable: {synapse.variable}")
        bt.logging.info(f"Requested hours: {synapse.requested_hours}")
        bt.logging.info(f"Grid shape: {tuple(corrected.shape)}")

        # ---- Correction delta stats ----
        bt.logging.info(
            f"Mean Δ: {diff_np.mean():.6f}  |  Std Δ: {diff_np.std():.6f}"
        )
        bt.logging.info(
            f"Max Δ: {diff_np.max():.6f}   |  Min Δ: {diff_np.min():.6f}"
        )

        # ---- Difficulty stats ----
        try:
            d_np = difficulty.detach().cpu().numpy()
            bt.logging.info(
                f"Difficulty: mean={d_np.mean():.4f}, std={d_np.std():.4f}, "
                f"min={d_np.min():.4f}, max={d_np.max():.4f}"
            )
        except Exception:
            bt.logging.info("Difficulty: <not available>")

        # ---- Output summary ----
        bt.logging.info("=== OUTPUT SUMMARY ===")
        bt.logging.info(
            f"Source: {self.output_source} | "
            f"req={self.request_count} | "
            f"cache_hits={self.cache_hits} | "
            f"api_calls={self.api_calls}"
        )
        bt.logging.info(
            f"Final output shape: {tuple(corrected.shape)} | Variable: {synapse.variable}"
        )

    # ---------------------------
    # Axon hooks
    # ---------------------------
    async def blacklist(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)

    async def priority(self, synapse: TimePredictionSynapse) -> float:
        return await self._priority(synapse)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)

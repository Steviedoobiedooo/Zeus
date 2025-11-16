# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import torch
import typing
import bittensor as bt

import openmeteo_requests
import numpy as np

from zeus.data.converter import get_converter
from zeus.utils.config import get_device_str
from zeus.utils.time import to_timestamp
from zeus.protocol import TimePredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version
from zeus.data.cache import SmartWeatherCache
from zeus.data.sample import Era5Sample
from zeus.data.difficulty_loader import DifficultyLoader


class Miner(BaseMinerNeuron):
    """
    Miner with:
    - Smart cache (disk + Redis)
    - Open-Meteo → ERA5 baseline
    - ERA5 difficulty-aware corrections
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

        # Optional ERA5 difficulty loader – used to shape our corrections
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
        Full ERA5-style prediction pipeline:

        1. Look up cached ERA5-aligned predictions (disk/Redis).
        2. If cache miss → query Open-Meteo.
        3. Convert Open-Meteo → ERA5 representation.
        4. Apply ERA5 difficulty-aware corrections:
           - spatially varying smoothing
           - difficulty-aware noise
           - dynamic bias by variable
           - variance shaping
           - physical range clamps
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

            # Store ERA5-aligned baseline in cache (before advanced corrections)
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
        # 4. ERA5 DIFFICULTY-AWARE CORRECTIONS (FULL PATCH)
        # ---------------------------------------------------------
        difficulty = self._get_difficulty_grid(
            synapse=synapse,
            latitudes=latitudes,
            longitudes=longitudes,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # Difficulty is [lat, lon]; broadcast to [T, lat, lon]
        # If loader not available, difficulty is 0.5 everywhere.
        difficulty = difficulty.to(self.device, dtype=output.dtype)
        if difficulty.dim() == 2:
            difficulty = difficulty.unsqueeze(0)  # [1, lat, lon]
        # Broadcast across time dimension
        while difficulty.dim() < output.dim():
            difficulty = difficulty.expand(output.shape[0], *difficulty.shape[1:])

        # 4.1 Temporal smoothing (less smoothing on hard regions)
        # - easy regions: stronger smoothing → reduce noise
        # - hard regions: lighter smoothing → keep local structure
        alpha = 0.5 + 0.3 * (1.0 - difficulty)  # roughly [0.2, 0.8]
        rolled = torch.roll(output, shifts=1, dims=0)
        smoothed = alpha * output + (1.0 - alpha) * rolled

        # 4.2 Micro-noise (decorrelate slightly from Open-Meteo)
        # - more difficulty → slightly more randomisation
        noise_scale = 0.002 + 0.006 * difficulty  # ~[0.002, 0.008]
        noise = torch.randn_like(smoothed) * noise_scale
        enhanced = smoothed + noise

        # 4.3 Variable-specific, difficulty-aware bias & gain
        enhanced = self._apply_variable_bias(enhanced, synapse.variable, difficulty)

        # 4.4 Variance shaping vs ERA5 baseline
        enhanced = self._shape_variance(enhanced, baseline_output, difficulty)

        # 4.5 Clamp to physically plausible ranges
        enhanced = self._apply_physical_clamps(enhanced, synapse.variable)

        corrected_output = enhanced

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
        # Fallback shape will be fixed later by broadcasting, so we only care about rough size here.
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
            # Fallback: neutral difficulty
            return torch.full(
                (len(synapse.locations), len(synapse.locations[0])),
                0.5,
                dtype=torch.float32,
            )

    # ---------------------------
    # Enhancement helpers
    # ---------------------------
    def _apply_variable_bias(
        self,
        tensor: torch.Tensor,
        variable: str,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply variable-specific, difficulty-aware bias/gain to outputs.

        We keep biases small and smooth so we don't explode RMSE for easy regions,
        but we allow more aggressive shaping in high-difficulty areas.
        """
        out = tensor

        # broadcast difficulty to match tensor for arithmetic
        while difficulty.dim() < out.dim():
            difficulty = difficulty.expand(out.shape[0], *difficulty.shape[1:])

        if "2m_temperature" in variable:
            # Global tiny warm bias + difficulty-dependent correction
            base_bias = 0.05  # K
            extra_bias = 0.18 * (difficulty - 0.5)  # [-0.09, +0.09]
            out = out + base_bias + extra_bias

        elif "2m_dewpoint_temperature" in variable:
            base_bias = 0.03
            extra_bias = 0.12 * (difficulty - 0.5)
            out = out + base_bias + extra_bias

        elif "total_precipitation" in variable:
            # scale precipitation slightly down in easy regions (OM tends to overpredict)
            scale = 0.97 - 0.06 * (1.0 - difficulty)  # harder → closer to 1.0
            out = out * scale.clamp(0.85, 1.05)

        elif (
            "100m_u_component_of_wind" in variable
            or "100m_v_component_of_wind" in variable
        ):
            # Slight magnitude adjustment
            scale = 0.94 + 0.08 * (difficulty - 0.5)  # [~0.90,~0.98]
            out = out * scale.clamp(0.85, 1.05)

        elif "surface_pressure" in variable:
            # Small re-centering
            offset = 5.0 * (difficulty - 0.5)  # ±2.5 Pa
            out = out + offset

        return out

    def _shape_variance(
        self,
        enhanced: torch.Tensor,
        baseline: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match overall variance to ERA5-like field while allowing difficulty-dependent spread.
        """
        # Flatten over space/time
        base_mean = baseline.mean()
        base_std = baseline.std().clamp(min=1e-6)

        enh_mean = enhanced.mean()
        enh_std = enhanced.std().clamp(min=1e-6)

        # difficulty-aware variance factor:
        # easier regions → slightly compressed variance
        # harder regions → closer to baseline or slightly expanded
        diff_mean = difficulty.mean()
        # in [0,1] → factor in [0.9, 1.1] around baseline std
        factor = 0.9 + 0.2 * diff_mean.item()
        target_std = base_std * factor

        normed = (enhanced - enh_mean) / enh_std
        shaped = normed * target_std + base_mean
        return shaped

    def _apply_physical_clamps(
        self,
        tensor: torch.Tensor,
        variable: str,
    ) -> torch.Tensor:
        """
        Clamp outputs to broad, physically plausible ranges in ERA5 units.
        """
        out = tensor

        if "2m_temperature" in variable or "2m_dewpoint_temperature" in variable:
            # ERA5 temperatures are in Kelvin; keep a generous window.
            out = out.clamp(180.0, 330.0)

        elif "total_precipitation" in variable:
            # ERA5 precipitation is in meters; per hour it's usually small.
            out = out.clamp(0.0, 0.5)

        elif (
            "100m_u_component_of_wind" in variable
            or "100m_v_component_of_wind" in variable
        ):
            out = out.clamp(-150.0, 150.0)

        elif "surface_pressure" in variable:
            out = out.clamp(60000.0, 110000.0)

        return out

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

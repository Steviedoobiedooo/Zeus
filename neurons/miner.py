# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

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


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to OpenMeteo (https://open-meteo.com/) for predictions.
    You are encouraged to attempt to improve over this by changing the forward function.
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

        # TODO(miner): Anything specific to your use case you can do here
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()

        # Hybrid cache: disk (historical) + Redis (forecast)
        self.weather_cache = SmartWeatherCache(
            disk_dir="/home/steve/projects/Zeus/.cache",
            max_bytes=100 * 1024 * 1024 * 1024,  # 100 GB
            redis_url="redis://127.0.0.1:6379",
            forecast_ttl_seconds=2 * 60 * 60,   # 2 hours
        )

        output_source = None

    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse for a prediction.
        """
        # increment total received requests
        self.request_count += 1

        coordinates = torch.Tensor(synapse.locations)
        start_time_dt = to_timestamp(synapse.start_time)
        end_time_dt = to_timestamp(synapse.end_time)

        bt.logging.info(
            f"[Request #{self.request_count}] "
            f"Predicting {synapse.requested_hours}h of {synapse.variable} "
            f"for grid {coordinates.shape}"
        )

        start_ts = float(synapse.start_time)
        end_ts = float(synapse.end_time)

        latitudes, longitudes = coordinates.view(-1, 2).T
        converter = get_converter(synapse.variable)

        coords_np = coordinates.numpy()

        # ---- Debug logging for cache analysis ----
        bt.logging.info(
            f"[CACHE DEBUG] variable={synapse.variable} | "
            f"start_ts={start_ts} | end_ts={end_ts} | "
            f"coords_shape={coords_np.shape}"
        )

        # ---- Try Cache ----
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
                f"(req #{self.request_count}) for {synapse.variable}"
            )

            output = torch.from_numpy(cached)

        else:
            # ---- Cache Miss → Open Meteo ----
            # ---- Cache Miss → Open Meteo ----
            self.api_calls += 1
            self.output_source = "Open Meteo"

            bt.logging.info(f"Cache MISS → Open-Meteo call #{self.api_calls}")

            params = {
                "latitude": latitudes.tolist(),
                "longitude": longitudes.tolist(),
                "hourly": converter.om_name,
                "start_hour": start_time_dt.isoformat(timespec="minutes"),
                "end_hour": end_time_dt.isoformat(timespec="minutes"),
            }

            # Add your paid API key
            api_key = os.getenv("OPEN_METEO_API_KEY")
            if api_key:
                params["apikey"] = api_key

            responses = self.openmeteo_api.weather_api(
                "https://customer-api.open-meteo.com/v1/forecast",
                params=params,
                method="POST",
            )

            output = torch.Tensor(
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
                )
            ).reshape(synapse.requested_hours, *coordinates.shape[:2], -1)

            output = output.squeeze(dim=-1)
            output = converter.om_to_era5(output)

            # store to cache
            self.weather_cache.set(
                variable=synapse.variable,
                start_time=start_ts,
                end_time=end_ts,
                coordinates=coords_np,
                data=output.cpu().numpy(),
            )

        # --- Log output summary ---
        bt.logging.info(
            f"Output shape {output.shape} | from {self.output_source} | "
            f"req={self.request_count} cache_hits={self.cache_hits} api_calls={self.api_calls}"
        )

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse

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

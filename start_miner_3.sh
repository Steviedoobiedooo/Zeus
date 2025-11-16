#!/bin/bash

set -a
source miner_3.env
set +a

MINER_PROCESS_NAME="zeus_miner_3"


if pm2 list | grep -q "$MINER_PROCESS_NAME"; then
  echo "Process '$MINER_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $MINER_PROCESS_NAME
fi

pm2 start neurons/miner.py --name $MINER_PROCESS_NAME -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.ip 0.0.0.0 \
  --axon.external_ip $AXON_EXTERNAL_IP \
  --axon.port $AXON_PORT \
  --axon.external_port $AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
  --logging.debug

# synchronise the process list with the pm2 ecosystem file
pm2 save
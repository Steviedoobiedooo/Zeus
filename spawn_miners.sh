#!/bin/bash

set -a
source spawn_miners.env
set +a

# Base config
PROCESS_BASE="zeus_miner"
PORT_BASE=${AXON_PORT_BASE:-8080}
COUNT=${MINER_COUNT:-10}

echo "🚀 Spawning $COUNT Zeus miners..."

for i in $(seq 1 $COUNT); do
    PROCESS_NAME="${PROCESS_BASE}_${i}"
    PORT=$((PORT_BASE + i - 1))

    # Optional: Automatically map hotkeys miner1-hotkey, miner2-hotkey, etc.
    HOTKEY_VAR="WALLET_HOTKEY_$i"
    HOTKEY_VALUE=${!HOTKEY_VAR:-$WALLET_HOTKEY}

    echo "➡️ Starting $PROCESS_NAME on port $PORT (hotkey: $HOTKEY_VALUE)"

    # Remove old instance
    if pm2 list | grep -q "$PROCESS_NAME"; then
        echo "🔄 Process $PROCESS_NAME already running — deleting..."
        pm2 delete "$PROCESS_NAME"
    fi

    pm2 start neurons/miner.py --name "$PROCESS_NAME" -- \
      --netuid $NETUID \
      --subtensor.network $SUBTENSOR_NETWORK \
      --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
      --wallet.name $WALLET_NAME \
      --wallet.hotkey $HOTKEY_VALUE \
      --axon.ip 0.0.0.0 \
      --axon.external_ip $AXON_EXTERNAL_IP \
      --axon.port $PORT \
      --axon.external_port $PORT \
      --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
      --logging.debug
done

pm2 save
echo "🎉 All $COUNT miners started & saved!"

#!/bin/bash
# FreeSWITCH Configuration Setup for sofia-phone

set -e

echo "🔧 Configuring FreeSWITCH for sofia-phone..."
echo "=============================================="

# FreeSWITCH config directory
FS_CONF="/opt/homebrew/etc/freeswitch"

# Check if FreeSWITCH is installed
if [ ! -d "$FS_CONF" ]; then
    echo "❌ FreeSWITCH not found. Run ./scripts/install_freeswitch_mac.sh first"
    exit 1
fi

echo "✅ FreeSWITCH config directory found: $FS_CONF"

# Backup existing configs
echo "📦 Creating backup of existing configs..."
BACKUP_DIR="$FS_CONF/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
if [ -f "$FS_CONF/sip_profiles/internal.xml" ]; then
    cp "$FS_CONF/sip_profiles/internal.xml" "$BACKUP_DIR/"
    echo "   Backed up: internal.xml"
fi

# Copy SIP profile
echo "📝 Installing SIP profile..."
cp config/freeswitch/sip_profiles/internal.xml "$FS_CONF/sip_profiles/"
echo "   → $FS_CONF/sip_profiles/internal.xml"

# Copy dialplan
echo "📝 Installing dialplan..."
mkdir -p "$FS_CONF/dialplan/public"
cp config/freeswitch/dialplan/sofia_phone.xml "$FS_CONF/dialplan/public/"
echo "   → $FS_CONF/dialplan/public/sofia_phone.xml"

echo ""
echo "✅ FreeSWITCH configured successfully!"
echo ""
echo "Next steps:"
echo "  1. Start FreeSWITCH:  freeswitch -nc -nonat"
echo "  2. Check status:      fs_cli"
echo "  3. Test with Zoiper:  Download from https://www.zoiper.com/en/voip-softphone/download/current"
echo ""
echo "Zoiper configuration:"
echo "  - Account type: SIP"
echo "  - Domain: $(ipconfig getifaddr en0 || echo "YOUR_MAC_IP")"
echo "  - Port: 5060"
echo "  - Username: (any)"
echo "  - Password: (leave empty - auth disabled for testing)"
echo ""

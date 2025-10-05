#!/bin/bash
# FreeSWITCH Installation Script for macOS

set -e

echo "🚀 Installing FreeSWITCH on macOS..."
echo "===================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"

# Update Homebrew
echo "📦 Updating Homebrew..."
brew update

# Install FreeSWITCH
echo "📥 Installing FreeSWITCH..."
brew install freeswitch

# Verify installation
if command -v freeswitch &> /dev/null; then
    echo "✅ FreeSWITCH installed successfully"
    freeswitch -version
else
    echo "❌ FreeSWITCH installation failed"
    exit 1
fi

# Get FreeSWITCH configuration directory
FS_CONF_DIR=$(brew --prefix)/etc/freeswitch

echo ""
echo "📁 FreeSWITCH configuration directory: $FS_CONF_DIR"
echo ""
echo "✅ Installation complete!"
echo ""
echo "To start FreeSWITCH:"
echo "  freeswitch -nc -nonat"
echo ""
echo "To test:"
echo "  fs_cli  (FreeSWITCH command line interface)"
echo ""

#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝                     

EOF

#This oveclocking script uses upp to tune clocks voltages and power limits.
#Use at your own risk!

MI50_POWER=225    #Power limit:do not go over 300W
MI50_SCLK=2000    #Core clock speed
MI50_MCLK=1100    #Memory clock speed
MI50_TDC_GFX=260  #TDC limit for GFX (default 330)
MI50_CARD=1       #For multi gpu config, choose your MI50!

DEVICE="/sys/class/drm/card${MI50_CARD}/device"
UPP_PYTHON="/home/iacoppbk/upp/bin/python3"

[ ! -f "$UPP_PYTHON" ] && echo "Error: UPP not found at $UPP_PYTHON" && exit 1

echo "Applying: Core=${MI50_SCLK}MHz, Mem=${MI50_MCLK}MHz, Power=${MI50_POWER}W, TDC_GFX=${MI50_TDC_GFX}"

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/SocketPowerLimitAc0=$MI50_POWER \
    smcPPTable/SocketPowerLimitDc=$MI50_POWER \
    smcPPTable/TdcLimitGfx=$MI50_TDC_GFX

sudo "$UPP_PYTHON" -m upp.upp -p ${DEVICE}/pp_table set --write \
    smcPPTable/FreqTableGfx/8=$MI50_SCLK \
    smcPPTable/FreqTableUclk/2=$MI50_MCLK \
    smcPPTable/FreqTableUclk/3=$MI50_MCLK

echo "Done"

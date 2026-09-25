#!/bin/bash

OBJDUMP=$1
ADDR2LINE=$2
TARGET_DEBUG=$3
TARGET_ASM=${TARGET_DEBUG}.S
BAD_INST_FILE=${TARGET_DEBUG}-BAD-INST.log

# grep expression to find unimplemented instructions
UNIMPLEMENTED_EXPR="fdiv.s\\|fsqrt.s\\|fcvt.l.s\\|fcvt.lu.s\\|fcvt.s.l\\|fcvt.s.lu\\|fdiv.pi\\|fdivu.pi\\|fremu.pi\\|frem.pi\\|fdiv.ps\\|fsqrt.ps\\|frsq.ps\\|fsin.ps"

# dump assembly into .S file
${OBJDUMP} -lwdSC ${TARGET_DEBUG} > ${TARGET_ASM}

# check with grep for unimplemented instructions
# Note: The exit status is 0 if selected lines are found, and 1 if not found.
grep ${UNIMPLEMENTED_EXPR} ${TARGET_ASM} > /dev/null
ret=$?

if [ ${ret} -eq 0 ]
then
    # unimplemented instructions are found
    echo -e "BUILD ERROR: Executable file ${TARGET_DEBUG} contains unimplemented instructions. Please review the lines of code listed in ${BAD_INST_FILE}"
    echo -e "\t     For further details, please read paragraph 3.4 of the ETSoC-1 Programmer's Reference Manual (PRM)"

    # addr2line
    grep ${UNIMPLEMENTED_EXPR} ${TARGET_ASM} | cut -d: -f 1 | ${ADDR2LINE} -i -e ${TARGET_DEBUG} > ${BAD_INST_FILE}
    grep ${UNIMPLEMENTED_EXPR} ${TARGET_ASM} >> ${BAD_INST_FILE}
    echo "------------------------------------------------------------"
    cat ${BAD_INST_FILE}
    echo "------------------------------------------------------------"
    exit 1

else
   rm -f ${BAD_INST_FILE}
fi

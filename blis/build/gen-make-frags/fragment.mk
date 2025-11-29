#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# fragment.mk 
#
# This is an automatically-generated makefile fragment and will likely get
# overwritten or deleted if the user is not careful. Modify at your own risk.
#

# These two mmakefile variables need to be set in order for the recursive
# include process to work!
CURRENT_DIR_NAME := _mkfile_fragment_cur_dir_name_
CURRENT_SUB_DIRS := _mkfile_fragment_sub_dir_names_

# Source files local to this fragment
LOCAL_SRC_FILES  := _mkfile_fragment_local_src_files_

# Add the fragment's local source files to the _global_variable_ variable.
_mkfile_fragment_src_var_name_ += $(addprefix $(PARENT_SRC_PATH)/$(CURRENT_DIR_NAME)/, $(LOCAL_SRC_FILES))




# -----------------------------------------------------------------------------
# NOTE: The code below is generic and should remain in all fragment.mk files!
# -----------------------------------------------------------------------------

# Add the current fragment to the global list of fragments so the top-level
# Makefile knows which directories are participating in the build.
FRAGMENT_DIR_PATHS  += $(PARENT_SRC_PATH)/$(CURRENT_DIR_NAME)

# Recursively descend into other subfragments' local makefiles and include them.
ifneq ($(strip $(CURRENT_SUB_DIRS)),)
key1                := $(key1).x
key2                := $(key2).y
stack_$(key1)       := $(PARENT_PATH)
stack_$(key2)       := $(PARENT_SRC_PATH)
PARENT_PATH         := $(PARENT_PATH)/$(CURRENT_DIR_NAME)
PARENT_SRC_PATH     := $(PARENT_SRC_PATH)/$(CURRENT_DIR_NAME)
FRAGMENT_SUB_DIRS   := $(addprefix $(PARENT_PATH)/, $(CURRENT_SUB_DIRS))
-include  $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_SUB_DIRS))
PARENT_PATH         := $(stack_$(key1))
PARENT_SRC_PATH     := $(stack_$(key2))
key1                := $(basename $(key1))
key2                := $(basename $(key2))
endif

SHELL=sh

# ******************************************************************************
#     ____ ___  _     ___  ____     __                  _   _                   _ _ _         
#    / ___/ _ \| |   / _ \|  _ \   / _|_   _ _ __   ___| |_(_) ___  _ __   __ _| (_) |_ _   _ 
#   | |  | | | | |  | | | | |_) | | |_| | | | '_ \ / __| __| |/ _ \| '_ \ / _` | | | __| | | |
#   | |__| |_| | |__| |_| |  _ <  |  _| |_| | | | | (__| |_| | (_) | | | | (_| | | | |_| |_| |
#    \____\___/|_____\___/|_| \_\ |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|\__,_|_|_|\__|\__, |
#                                                                                       |___/ 
RESET          = \033[0m
DIMM           = \e[2mDim
make_std_color = \033[3$1m      # defined for 1 through 7
make_color     = \033[38;5;$1m  # defined for 1 through 255
WRN_COLOR = $(strip $(call make_std_color,3))
ERR_COLOR = $(strip $(call make_std_color,1))
TS1_COLOR = $(strip $(call make_std_color,1)) #RED
TS2_COLOR = $(strip $(call make_std_color,2)) #GREEN
TS3_COLOR = $(strip $(call make_std_color,3)) #ORANGE
TS4_COLOR = $(strip $(call make_std_color,4)) #BLUE
TS5_COLOR = $(strip $(call make_std_color,5)) #PURPLE
TS6_COLOR = $(strip $(call make_std_color,6)) #TURQUISE
TS7_COLOR = $(strip $(call make_std_color,7)) #PLAIN
STD_COLOR = $(strip $(call make_color,8))

COLOR_OUTPUT = 2>&1 |                                   \
    while IFS='' read -r line; do                       \
        if  [[ $$line == *ERROR* ]]; then         		\
            echo -e "$(ERR_COLOR)$${line}$(RESET)";     \
        elif [[ $$line == *warning* ]]; then      		\
            echo -e "$(WRN_COLOR)$${line}$(RESET)";     \
        elif [[ $$line == Loading* ]]; then      		\
            echo -e "$(TS6_COLOR)$${line}$(RESET)";     \
        elif [[ $$line == Testing* ]]; then      		\
            echo -e "$(TS2_COLOR)$${line}$(RESET)";     \
        else                                            \
            echo -e "$(STD_COLOR)$${line}$(RESET)";     \
        fi;                                             \
    done; exit $${PIPESTATUS[0]};


GIT_BRANCH_OUT  = $(shell git status b|grep "On branch"| awk '{ print $3 }')
GIT_BRANCH = $(subst On branch ,,$(GIT_BRANCH_OUT))

# SOURCE      := $(wildcard src/*.cpp) $(foreach mod, $(MODULES),$(wildcard src/$(mod)/*.cpp))
# TEMP        := $(subst src/,obj/,$(SOURCE))
# OBJS        := $(subst .cpp,.o,$(TEMP))
# HEADERS     := $(wildcard inc/*.h

# Define the list of tasks you want to run
TASKS := 1 2 3 4

.PHONY: test

all: lab1

push:
	@git push github master

# Use LAB as a variable that can be set when calling make
build_lab%:
	@echo -e "$(TS2_COLOR)-= Building lab_$* =-$(RESET)";	
	@make --no-print-directory -C lab_$*

lab%: build_lab%
	@echo -e "$(TS4_COLOR)-= Running lab_$* =-$(RESET)";	
	@#cd lab_$* && ./lab$*
	@for task in $(TASKS); do \
		echo -e "$(TS6_COLOR)-= Running task_$${task} =-$(RESET)"; \
		echo -e "$(TS5_COLOR)| cd lab_$* && ./lab$* $${task} |$(RESET)"; \
		cd lab_$* && ./lab$* -t $${task}; \
		cd ..; \
	 done

lab_%:
	@make lab$*

# Special rule for lab0
build_lab0:
	@echo "Building lab_0 folder1"
	@make --no-print-directory -C lab_0/pc

lab0: build_lab0
	@echo "Running lab_0 pc"
	@cd lab_0/pc && ./lab0  # or whatever the executable/script is

test: $(ALL_TEST_RESULTS)
	@echo -e "$(TS2_COLOR)"; #//GREEN
	@echo "###################################################"
	@echo "#    _____         _   _                          #"
	@echo "#   |_   _|__  ___| |_(_)_ __   __ _              #"
	@echo "#     | |/ _ \/ __| __| | '_ \ / _\` |             #"
	@echo "#     | |  __/\__ \ |_| | | | | (_| |             #"
	@echo "#     |_|\___||___/\__|_|_| |_|\__, |             #"
	@echo "#                              |___/              #"
	@echo "#     ____                      _      _          #"
	@echo "#    / ___|___  _ __ ___  _ __ | | ___| |_ ___    #"
	@echo "#   | |   / _ \| '_ \` _ \| '_ \| |/ _ \ __/ _ \   #"
	@echo "#   | |__| (_) | | | | | | |_) | |  __/ ||  __/   #"
	@echo "#    \____\___/|_| |_| |_| .__/|_|\___|\__\___|   #"
	@echo "#                        |_|                      #"
	@echo "###################################################"
	@echo "# Testing Performed on $(HOSTNAME) as $(USER)"
	@echo "###################################################"
	@echo -e "# Testing Complete!"; #//GREEN
	@echo "###################################################"
	@echo "# DEBUG $(ALL_MODULES) as $(USER)"
	@echo "###################################################"
	@echo -e "$(RESET)"; #//GREEN

clean:
	@/bin/rm -rf result

debug:
	@#echo $(ALL_MODULES);
	@echo -e "$(TS1_COLOR)Folders or Modules$(RESET)"; #//RED
	@echo -e "$(TS2_COLOR)Folders or Modules$(RESET)"; #//GREEN
	@echo -e "$(TS3_COLOR)Folders or Modules$(RESET)"; #//ORANGE
	@echo -e "$(TS4_COLOR)Folders or Modules$(RESET)"; #//BLUE
	@echo -e "$(TS5_COLOR)Folders or Modules$(RESET)"; #//PURPLE
	@echo -e "$(TS6_COLOR)Folders or Modules$(RESET)"; #//TURQOISE
	@echo -e "$(TS7_COLOR)Folders or Modules$(RESET)"; #//PLAIN
	@#echo "Folders or Modules" $(COLOR_OUTPUT)       #//DIMM

	@#echo $(subst "\s+ ","\n"/,$(ALL_TEST_RESULTS))

stat:
	@echo "Status of Debug: $(USER)"
	@echo "USER: $(USER)"
	@echo "HOSTNAME: $(HOSTNAME)"
	@echo "HOSTTYPE: $(HOSTTYPE)"
	@echo "OSTYPE: $(OSTYPE)"
	@echo "TESTING THIS FOR THE LAST TIME"

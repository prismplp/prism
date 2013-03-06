# -*- Makefile -*-

ROOT = .

include $(ROOT)/makefiles/Makefile.opts.gmake
include $(ROOT)/makefiles/Makefile.files

##----------------------------------------------------------------------

ifeq ($(PROCTYPE),mp)
SUBDIRS += $(MP_DIR)
OBJS    += $(MP_OBJS)
endif

##----------------------------------------------------------------------

all: $(TARGET)

install: $(TARGET)
	$(CP) $(TARGET) $(INSTALLDIR)

clean: clean_subdirs
	$(RM) $(TARGET)

clean_subdirs:
	for i in $(SUBDIRS); do                     \
	  ($(MAKE) -f $(MAKEFILE) -C $$i clean )    \
	done
core:
	cd $(CORE_DIR) && $(MAKE) -f $(MAKEFILE) all

up:
	cd $(UP_DIR) && $(MAKE) -f $(MAKEFILE) all

mp:
	if [ $(PROCTYPE) = mp ]; then                 \
	  (cd $(MP_DIR) && $(MAKE) -f $(MAKEFILE) all)  \
	fi

$(TARGET): $(SUBDIRS) $(BP4P_A)
	$(CC) $(TARGETFLAG) $(TARGET) $(OBJS) $(BP4P_A) $(LDFLAGS)

##----------------------------------------------------------------------

.PHONY: all install clean $(SUBDIRS)

##----------------------------------------------------------------------

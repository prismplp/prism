# -*- Makefile -*-

ROOT = .

include $(ROOT)/makefiles/Makefile.opts.gmake

ifeq ($(MATRIX),lapack)
include $(ROOT)/makefiles/Makefile_lapack.files
else
include $(ROOT)/makefiles/Makefile.files
endif

ifeq ($(PLATFORM),linux64)
ifeq ($(MATRIX),lapack)
## this order is important
LAPACK_OBJS = lapack$(S)liblapacke.$(A)    \
            lapack$(S)liblapack.$(A)   \
            lapack$(S)libtmglib.$(A)    \
            lapack$(S)librefblas.$(A)      
endif
endif
ifeq ($(PLATFORM),linux32)
ifeq ($(MATRIX),lapack)
LAPACK_OBJS = lapack32$(S)liblapacke.$(A)    \
            lapack32$(S)liblapack.$(A)   \
            lapack32$(S)libtmglib.$(A)    \
            lapack32$(S)librefblas.$(A)      
endif
endif



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
	$(LD) $(TARGETFLAG) $(TARGET) $(OBJS) $(BP4P_A) $(LDFLAGS)

##----------------------------------------------------------------------

.PHONY: all install clean $(SUBDIRS)

##----------------------------------------------------------------------

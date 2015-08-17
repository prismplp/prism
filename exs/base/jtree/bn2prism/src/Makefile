ENCODE = UTF-8
DOC_DIR = ../doc/html
LIB_DIR = ../lib
CLASSPATH = $(LIB_DIR)
#MEMSIZE = 512m
#FLAGS = -d64

JAVA_FILES = BN2Prism.java BNSetting.java BayesNet.java Cluster.java ClusterNode.java \
             CPTNode.java InstanciateNode.java Format.java B2PException.java \
             B2PHandler.java Variable.java VArray.java package-info.java

CLASS_FILES = BN2Prism.class BNSetting.class BayesNet.class Cluster.class ClusterNode.class \
              CPTNode.class InstanciateNode.java Format.class B2PException.class \
              B2PHandler.class Variable.class VArray.class

################################################################

.PHONY: all class doc clean cleanall cleandoc

all: class

doc:
	javadoc -classpath $(CLASSPATH) -d $(DOC_DIR) -private -encoding $(ENCODE) $(JAVA_FILES)

class:
	javac $(FLAGS) -classpath $(CLASSPATH) -d $(LIB_DIR) -encoding $(ENCODE) $(JAVA_FILES)

cleanall: clean cleandoc

clean:
	find $(LIB_DIR) -name '*.class' -exec rm -f {} \;

cleandoc:
	rm -rf $(DOC_DIR)

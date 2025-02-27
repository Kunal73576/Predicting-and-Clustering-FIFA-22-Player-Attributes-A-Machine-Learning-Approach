export HADOOP_VERSION=3.3.6
export HADOOP_HOME=$HOME/Hadoop/hadoop-$HADOOP_VERSION
export PATH=${PATH}:$HADOOP_HOME/bin
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/opt/openblas/lib

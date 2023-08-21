# AI Stroke Diagnosis Data Pipeline: NiFi Python Processors

First we go over how to build NiFi 2.0 Snapshot with Python Processor extensibility and then we deploy NiFi and then we go over each of our custom **sjsu_ms_ai** NiFi Python Processors for AI Stroke Diagnosis.

## Build NiFi 2.0 Snapshot with Python Processors

~~~bash
# conda create --name stroke_ai_nifi -c conda-forge openjdk==17.0.7.4 maven==3.9.2 python==3.9.17
conda create --name stroke_ai_nifi python==3.9.17

sudo apt -y install openjdk-17-jdk

# Download maven 3.9.2+
# go: https://maven.apache.org/download.cgi

# get bin/java path
which java

# Get alt java paths
update-alternatives --list java

# Follow the steps here to install maven
# go: https://phoenixnap.com/kb/install-maven-on-ubuntu

source /etc/profile.d/maven.sh

# conda environment path should be java_home
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/

export PYTHON_HOME=/home/james/miniconda3/envs/stroke_ai_nifi/

# Environment variables:
export MAVEN_COMMAND="./mvnw"

export DEFAULT_MAVEN_OPTS="-Xmx3g -XX:ReservedCodeCacheSize=1g -XX:+UseG1GC -Dorg.slf4j.simpleLogger.defaultLogLevel=WARN -Daether.connector.http.retryHandler.count=5 -Daether.connector.http.connectionMaxTtl=30"

export COMPILE_MAVEN_OPTS="-Xmx3g -Daether.connector.http.retryHandler.count=5 -Daether.connector.http.connectionMaxTtl=30"

export MAVEN_COMPILE_COMMAND="test-compile --show-version --no-snapshot-updates --no-transfer-progress --fail-fast -pl -:minifi-c2-integration-tests -pl -:minifi-integration-tests -pl -:minifi-assembly -pl -:nifi-assembly -pl -:nifi-kafka-connector-assembly -pl -:nifi-kafka-connector-tests -pl -:nifi-toolkit-encrypt-config -pl -:nifi-toolkit-tls -pl -:nifi-toolkit-assembly -pl -:nifi-registry-assembly -pl -:nifi-registry-toolkit-assembly -pl -:nifi-runtime-manifest -pl -:nifi-runtime-manifest-test -pl -:nifi-stateless-assembly -pl -:nifi-stateless-processor-tests -pl -:nifi-stateless-system-test-suite -pl -:nifi-system-test-suite -pl -:nifi-nar-provider-assembly -pl -:nifi-py4j-integration-tests"

# Remove --no-snapshot-updates
export MAVEN_COMPILE_COMMAND_V2="test-compile --show-version --no-transfer-progress --fail-fast -pl -:minifi-c2-integration-tests -pl -:minifi-integration-tests -pl -:minifi-assembly -pl -:nifi-assembly -pl -:nifi-kafka-connector-assembly -pl -:nifi-kafka-connector-tests -pl -:nifi-toolkit-encrypt-config -pl -:nifi-toolkit-tls -pl -:nifi-toolkit-assembly -pl -:nifi-registry-assembly -pl -:nifi-registry-toolkit-assembly -pl -:nifi-runtime-manifest -pl -:nifi-runtime-manifest-test -pl -:nifi-stateless-assembly -pl -:nifi-stateless-processor-tests -pl -:nifi-stateless-system-test-suite -pl -:nifi-system-test-suite -pl -:nifi-nar-provider-assembly -pl -:nifi-py4j-integration-tests"


export MAVEN_VERIFY_COMMAND="verify --show-version --no-snapshot-updates --no-transfer-progress --fail-fast -D dir-only"

export MAVEN_BUILD_PROFILES="-P include-grpc -P skip-nifi-bin-assembly"

export MAVEN_PROJECTS="-pl -minifi/minifi-assembly
    -pl -minifi/minifi-c2/minifi-c2-assembly
    -pl -minifi/minifi-toolkit/minifi-toolkit-assembly
    -pl -nifi-registry/nifi-registry-assembly
    -pl -nifi-registry/nifi-registry-toolkit/nifi-registry-toolkit-assembly
    -pl -nifi-stateless/nifi-stateless-assembly
    -pl -nifi-toolkit/nifi-toolkit-assembly"

# Setup Java 17 (Static Analysis) - SUCCEED
$MAVEN_COMMAND validate --no-snapshot-updates --no-transfer-progress --fail-fast -P contrib-check -P include-grpc

# Setup Java 17 (Ubuntu Build-En Maven Compile) - SUCCEED
export MAVEN_OPTS=$COMPILE_MAVEN_OPTS
# $MAVEN_COMMAND $MAVEN_COMPILE_COMMAND
# $MAVEN_COMMAND $MAVEN_COMPILE_COMMAND_V2

$MAVEN_COMMAND clean install -T2C
~~~



## Deploy NiFi 2.0 Snapshot

~~~bash
cd nifi-assembly
ls -lhd target/nifi*

mkdir ~/src/forked/nifi-deploy-200-snapshot
unzip target/nifi-*-bin.zip -d ~/src/forked/nifi-deploy-200-snapshot

cd ~/src/forked/nifi-deploy-200-snapshot/nifi-*

./bin/nifi.sh start

./bin/nifi.sh start --wait-for-init 120

./bin/nifi.sh stop


# Check URL
https://localhost:8443/nifi

# NiFis UI URL in NiFi Properties:
/home/james/src/forked/nifi-deploy-200-snapshot/nifi-2.0.0-SNAPSHOT/conf/nifi.properties


# Find the random generated credentials
# cd logs/nifi-app.log
grep Generated logs/nifi-app*log

"""
Generated Username [04804a59-7c21-4f01-b58c-8f3d2b274226]
Generated Password [v2Pf+39lpZOFdrvBl+a6FpU1XDZhfMl+]
"""

# Update the login NiFi credentials

./bin/nifi.sh set-single-user-credentials james sjsu_nifi_123

~~~



# Outline

- NiFi Py Pipeline (Data Preparation Pipeline (SimpleITK))
- NiFi Py Pipeline (Preprocess MRI Stroke for Segmentation (ITK))
- NiFi Py Pipeline (Preprocess MRI Stroke for Text Generation (Spacy))
- NiFi Py Pipeline (Deploy Stroke Seg to 3D Slicer (PyTorch, PyIGTL))
- NiFi Py Pipeline (Deploy Stroke Clinical Gen to 3D Slicer (PyTorch, PyIGTL))

- Train MRI Stroke Segmentation (PyTorch)
- Train MRI Stroke Img Captions (PyTorch)
- Appendix

# NiFi Py Pipelines

## NiFi Py Data Preparation Pipeline

### GetNFBSNIfTIFiles

### GetICPSR38464NIfTIFiles

### GetATLASNIfTIFiles

### CorrectBiasFieldInITKImage

### ResizeCropITKImage

### SaveITKImageSlice


# Appendix

## NiFi Py4J 

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-py4j-integration-tests/src/test/java/org.apache.nifi.py4j/PythonControllerInteractionIT.java

## Python Processor Examples

target/python/extensions/WriteMessage.py



/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/ConvertCsvToExcel.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/DetectObjectInImage.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/GenerateRecord.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/LogContents.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/LookupAddress.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/PopulateRecord.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/PrettyPrintJson.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/SetRecordField.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/WriteMessage.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/WriteMessageV2.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/WriteNumpyVersion.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/WritePropertyToFlowFile.py

/home/james/src/forked/nifi/nifi-nar-bundles/nifi-py4j-bundle/nifi-python-test-extensions/src/main/resources/extensions/multi-module/WriteNumber.py

/home/james/src/forked/nifi/nifi-system-tests/nifi-system-test-suite/src/test/resources/conf/clustered/node1/nifi.properties

/home/james/src/forked/nifi/nifi-system-tests/nifi-system-test-suite/src/test/resources/conf/clustered/node2/nifi.properties




src/test/resources/json/input/simple-person.json

## References

- https://linux.how2shout.com/steps-to-install-openjdk-17-ubuntu-linux-such-as-22-04-or-20-04/
- https://phoenixnap.com/kb/install-maven-on-ubuntu
- https://maven.apache.org/download.cgi
- https://stackoverflow.com/questions/12787757/how-to-use-the-command-update-alternatives-config-java

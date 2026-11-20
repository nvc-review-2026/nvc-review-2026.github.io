#!/usr/bin/env bash
# Build the CTC encoders at the versions of the Isambard decoders (from their version strings):
#   VTM 23.13, ECM 19.0, libaom 3.13.1-213-gb6c94429a9, AVM 13.0.0-85-g1fd2cbbc1e
# into $NVC_ROOT/codecs/encoders/<codec>/ (sources + build on E:).
set -e
R=${NVC_ROOT:-VideoCompression}/codecs/encoders
mkdir -p $R && cd $R
J=$(nproc)

[ -d vtm ] || git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
(cd vtm && git checkout VTM-23.13 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$J)

[ -d ecm ] || git clone https://vcgit.hhi.fraunhofer.de/ecm/ECM.git ecm
(cd ecm && git checkout ECM-19.0 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$J)

[ -d av1 ] || git clone https://aomedia.googlesource.com/aom av1
(cd av1 && git checkout b6c94429a9 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=0 -DENABLE_DOCS=0 && cmake --build build -j$J)

[ -d avm ] || git clone https://gitlab.com/AOMediaCodec/avm.git avm
(cd avm && git checkout 1fd2cbbc1e && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=0 -DENABLE_DOCS=0 && cmake --build build -j$J)

ls -l vtm/bin/EncoderAppStatic ecm/bin/EncoderAppStatic av1/build/aomenc avm/build/{aomenc,avmenc} 2>&1

TEST_URL=http://www.csie.ntu.edu.tw/~cyy/courses/vfx/12spring/assignments/proj2/data/parrington.zip\
	http://www.csie.ntu.edu.tw/~cyy/courses/vfx/12spring/assignments/proj2/data/grail.zip\
	http://www.csie.ntu.edu.tw/~cyy/courses/vfx/12spring/assignments/proj2/data/denny.zip\
	http://www.csie.ntu.edu.tw/~cyy/courses/vfx/12spring/assignments/proj2/data/csie.zip

.PHONY: download
download:
	@cd test;\
	for url in ${TEST_URL}; do \
		wget $$url; \
	done;\
	unzip "*.zip";\
	rm *.zip

.PHONY: demo
demo:
	cd src && make
	./src/astich -f 11 test/demo/*

PACK_NAME := B97902018_B97902058
.PHONY: pack
pack:
	rm -rf ${PACK_NAME}
	mkdir -p ${PACK_NAME}/program
	cp -rf src/astich.cpp src/Makefile ${PACK_NAME}/program
	cp sift/sift_lib.cpp sift/sift_lib.h ${PACK_NAME}/program
	cd ${PACK_NAME}/program && make
	mkdir -p ${PACK_NAME}/test/original
	cp -rf test/demo/* ${PACK_NAME}/test/original
	mkdir -p ${PACK_NAME}/test/matched
	mkdir -p ${PACK_NAME}/test/stitched
	cp -f report/INSTRUCTION ${PACK_NAME}/
	cp -rf report ${PACK_NAME}
	mkdir -p ${PACK_NAME}/artifact/original
	cp -rf test/maca_small/* ${PACK_NAME}/artifact/original
	mkdir -p ${PACK_NAME}/artifact/stiched
	cp -rf report/img/maca/result.png ${PACK_NAME}/artifact/stiched
	./${PACK_NAME}/program/astich -f 11 ${PACK_NAME}/test/original/* -o ${PACK_NAME}/test/stiched/result \
		> ${PACK_NAME}/test/matched/matched.txt
	tar czvf ${PACK_NAME}.tgz ${PACK_NAME}

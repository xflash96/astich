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
	mkdir -f ${PACK_NAME}
	cp -rf src ${PACK_NAME}/program
	cd ${PACK_NAME}/program && make
	mkdir -f ${PACK_NAME}/test
	cp -rf test/demo ${PACK_NAME}/test/original
	mkdir -f ${PACK_NAME}/test/matched
	mkdir -f ${PACK_NAME}/test/stitched
	mkdir -f ${PACK_NAME}/instruction
	cp -rf report ${PACK_NAME}
	mkdir -f ${PACK_NAME}/artifact
	cp -rf test/maca_small/* ${PACK_NAME}/artifact/original
	mkdir -f ${PACK_NAME}/artifact/stiched
	cp -rf report/img/maca/result.png ${PACK_NAME}/artifact/stiched

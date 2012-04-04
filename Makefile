.PHONY: text
	echo 'hello world'

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
	
.PHONY: fetch
fetch:
	git submodule init
	git submoudle update

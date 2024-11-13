build:
	echo "built"
do_index:
	bash ./index_build ${INDEX_ARGS}
do_scan:
	bash ./index_scan ${SCAN_ARGS}1
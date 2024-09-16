setup:
	./helpers/setup.sh

nuitka:
	./helpers/package.sh

debug:
	./helpers/debug.sh

windows:
	./helpers/windows.sh

run:
	cd launch.dist && ./launch.exe
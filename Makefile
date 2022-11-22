install_dev:
	@pip install .[dev]

install:
	@pip install .

clean:
	@rm -f */version.txt
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build
	@rm -fr ZFF_VAD-*.dist-info
	@rm -fr ZFF_VAD.egg-info

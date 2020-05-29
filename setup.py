from setuptools import setup, find_packages

setup(
	name='genomics',
	version='0.0',
	description="""A Python project""",
	
	long_description="long_description",
	long_description_content_type="text/markdown",
	
	# please, change this for your project
	
	author='Kenenbek Arzymatov',
	author_email='kenenbek@gmail.com',
	
	maintainer='Kenenbek Arzymatov',
	maintainer_email='kenenbek@gmail.com',
	
	license='MIT',
	
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3',
	],
	
	packages=find_packages(where='src/', ),
	package_dir={'': 'src/'},
	
	extras_require={
		'test': [
			'pytest >= 4.0.0',
		],
	},
	
)

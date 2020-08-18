FROM continuumio/miniconda:latest

RUN conda install pytorch torchvision -c pytorch \
										 numpy \
										 matplotlib \
										 -c conda-forge tqdm

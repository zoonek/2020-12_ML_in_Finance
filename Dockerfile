FROM pytorch/pytorch:latest
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential wget r-base libcurl4-openssl-dev libssl-dev
RUN R -e 'install.packages(c("dplyr","data.table","tidyverse","xgboost","plyr"))'
RUN R -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/cowplot/cowplot_0.9.4.tar.gz")'
COPY    . /app
WORKDIR /app
RUN pip install -r requirements.txt

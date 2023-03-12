FROM ubuntu

RUN apt-get update &&\
	apt-get install -y python3.9 &&\
	apt-get install -y python3-pip &&\
	apt-get install unzip &&\
	mkdir /apps
	

RUN pip install xgboost &&\
	pip install gradio &&\
	pip install pandas &&\
	pip install scikit-learn &&\
	pip install numpy

COPY tout_pour_gradio.zip /apps/

RUN unzip /apps/tout_pour_gradio.zip -d /apps

RUN chmod 777 -R /apps

EXPOSE 7860



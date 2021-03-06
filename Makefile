CONTAINER = jmlehrer/organoids

exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

job:
	kubectl create -f organoid_growth_curves.yaml

retry:
	docker build -t $(CONTAINER) .
	docker push $(CONTAINER)
	kubectl delete job julians-experiment 
	kubectl create -f pod.yaml

cleancluster:
	rm -rf results/*
	aws --endpoint https://s3.nautilus.optiputer.net s3 rm s3://braingeneersdev/jlehrer/cluster_test/ --recursive


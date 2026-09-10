FROM tensorflow/serving:latest

# Copy folder model TF Serving ke dalam container
COPY ./serving_model/fraud_detection /models/fraud-detection
COPY ./monitoring /model_config

ENV MODEL_NAME=fraud-detection
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
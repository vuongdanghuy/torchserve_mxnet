TARGET = resnet50
MODEL_STORE = model_store
VERSION = 1.0
HANDLER = handler.py

override FLAGS += --extra-files src -f

all: $(MODEL_STORE)/$(TARGET).mar;

$(MODEL_STORE)/$(TARGET).mar: $(HANDLER) makefile
	torch-model-archiver --model-name $(TARGET) \
		--handler $(HANDLER) \
		--export-path $(MODEL_STORE) \
		-v $(VERSION) \
		$(FLAGS)

.PHONY: run
run:
	torchserve --start \
		--ncs \
		--ts-config config.properties \
		--model-store $(MODEL_STORE)

.PHONY: stop
.PHONY: clean
stop clean:
	torchserve --stop

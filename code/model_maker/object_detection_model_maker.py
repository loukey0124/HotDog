import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

os.environ["CUDA_VISIBLE_DEVICES"]="0"

train_data = object_detector.DataLoader.from_pascal_voc(
    './data/train', './data/train', ['dog', 'cat'])

val_data = object_detector.DataLoader.from_pascal_voc(
    './data/val', './data/val', ['dog', 'cat']
)

spec = model_spec.get('efficientdet_lite0')

model = object_detector.create(train_data=train_data, model_spec=spec, validation_data=val_data, batch_size=4, train_whole_model=True, epochs=10)

config = QuantizationConfig.for_float16()
model.export(export_dir='./', export_format=[ExportFormat.LABEL, ExportFormat.TFLITE],
             quantization_config=config)

test_data = object_detector.DataLoader.from_pascal_voc(
    './data/test', './data/test', ['dog', 'cat']
)

print(model.evaluate(test_data))

print(model.evaluate_tflite('model.tflite', test_data))
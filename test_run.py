from runner_builders.crisis_mmd_runners import get_crisis_mmd_vision_text_pipline_model_runners
from enums import SplitRunType
from utils import metric_utils, config_utils
# from lime.lime_text import LimeTextExplainer
from data.schemas import DataItemSchema
import constants
import json
from pathlib import Path

config = config_utils.load_config()

if __name__ == "__main__":
    test_model_runner_interface = get_crisis_mmd_vision_text_pipline_model_runners(
        0, True
    )[SplitRunType.TEST]

    captions = []
    image_paths = []
    image_ids = []

    data_loader = test_model_runner_interface.dataloader
    i = 0
    for batch_item in data_loader:
        i+=1
        if i==30:
            break

        data_item = DataItemSchema(**batch_item)

        captions.extend(list(data_item.metadata['caption']))
        image_paths.extend(list(data_item.metadata['image_path']))
        image_ids.extend(list(data_item.metadata['image_id']))

    epoch_result = test_model_runner_interface.run_epoch(0)

    probabilities = epoch_result['probabilities']
    predictions = epoch_result['predictions']
    targets = epoch_result['targets']

    assert predictions.keys() == targets.keys()
    for k in predictions.keys():
        assert len(probabilities[k]) == len(predictions[k]) == len(targets[k]) == len(captions) == len(image_ids) == len(image_paths)
        metrics_dict = metric_utils.get_metrics(
            probabilities.get(k), predictions[k], targets[k]
        )
        print('metrics_dict', metrics_dict)


    # create a list of dictionary of {'image_id':, 'image_path':, 'caption':, 'image_path': 'caption', 'prediction':, 'target':}
    results = []
    for i in range(len(image_ids)):
        for k in predictions.keys():
            results.append({
                'k': k,
                'image_id': str(image_ids[i]),
                'image_path': str(image_paths[i]),
                'caption': str(captions[i]),
                'prediction': int(predictions[k][i]),
                'target': int(targets[k][i]),
                'is_correct': int(predictions[k][i]) == int(targets[k][i])
            })

    print('results', results)

    # save the results to a json file, create a file if it doesn't exist
    runner_identifier = test_model_runner_interface.get_runner_identifier()
    output_dir = Path("test_run_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{runner_identifier}_predictions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f)
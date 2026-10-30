"""Training/inference helpers shared by the new entry points."""
import torch
from display import memory_postfix, tqdm
from af_backend import training_utilities


def seed_everything(seed):
    return training_utilities().fix_random_seed(seed,include_cuda=torch.cuda.is_available())


@torch.no_grad()
def infer(model, dataset, cached=None, on_video=None, show_progress=True, description="Validation"):
    model.eval()
    results = dict(cached or {})
    print("******** Start Testing. ********",flush=True)
    remaining = [i for i,r in enumerate(dataset.records) if r["video_path"] not in results]
    with tqdm(remaining,total=len(dataset),initial=len(results),desc=description,unit="video",
              dynamic_ncols=True,disable=not show_progress) as progress:
        for number,index in enumerate(progress):
            video = dataset[index]
            output = model([video])[0]
            results[video["video_id"]] = dict(segments=output["segments"].tolist(),
                                             scores=output["scores"].tolist())
            if on_video is not None:
                on_video(video["video_id"], results[video["video_id"]])
            if number == 0 or (number+1) % 50 == 0 or number+1 == len(remaining):
                progress.set_postfix(memory_postfix())
    return results


def save_torch(path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

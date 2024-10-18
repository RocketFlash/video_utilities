import click
from pathlib import Path
from src.video_downloader import VideoDownloader


@click.command()
@click.option(
    "--video_ids", 
    type=str, 
    required=True, 
    multiple=True
)
@click.option(
    "--quality", 
    type=str, 
    default='max'
)
@click.option(
    "--url_template", 
    type=str, 
    required=True,
)
@click.option(
    "--secret", 
    type=str, 
    required=True,
)
@click.option(
    "--save_dir",
    default='./',
    type=str,
    required=True,
)
def download(
    video_ids,
    quality,
    url_template,
    secret,
    save_dir
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=save_dir,
    )

    video_downloader(video_ids)


if __name__ == '__main__':
    download()
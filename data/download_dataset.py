import roboflow


def main():
    rf = roboflow.Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("koloboktyresegmentation").project("tire-spikes-det-gbicv")
    version = project.version(4)
    version.download("coco-mmdetection")


if __name__ == "__main__":
    main()

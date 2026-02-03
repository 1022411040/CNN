import argparse
import asyncio
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Dynamic Image Classification System")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--crawl", type=str, help="Crawl images for a class")
    parser.add_argument("--add-class", type=str, help="Add a new class")
    parser.add_argument("--train", action="store_true", help="Train a new model")

    args = parser.parse_args()

    if args.serve:
        # ❗ DO NOT use asyncio here
        uvicorn.run(
            "inference_service:app",
            host="127.0.0.1",
            port=8000,
            workers=1,
            reload=False
        )
        return

    # Everything below uses asyncio safely
    asyncio.run(run_async_commands(args))


async def run_async_commands(args):
    if args.crawl:
        from image_crawler import ImageCrawler
        crawler = ImageCrawler()
        await crawler.crawl_class(args.crawl, max_images=500)
        print(f"Crawling completed for {args.crawl}")

    elif args.add_class:
        from class_registry import ClassRegistry
        registry = ClassRegistry()
        class_id = await registry.add_class(args.add_class)
        print(f"Class '{args.add_class}' added with ID: {class_id}")

    elif args.train:
        from trainer import train_new_model
        from class_registry import ClassRegistry
        from dynamic_model import ModelManager

        registry = ClassRegistry()
        model_manager = ModelManager()

        model_path = await train_new_model(
            dataset_version="latest",
            registry=registry,
            model_manager=model_manager
        )
        print(f"Training completed. Model saved to: {model_path}")

    else:
        print("Please specify an action: --serve, --train, --crawl, or --add-class")


if __name__ == "__main__":
    main()

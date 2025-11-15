import os
from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    print(DeepSeek_API_KEY)  # 可以通过打印查看


if __name__ == "__main__":
    main()

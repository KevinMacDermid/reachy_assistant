from src.wakeword import listen_for_wakeword

def main():
    listen_for_wakeword()
    print("Wakeword Received!")


if __name__ == '__main__':
    main()


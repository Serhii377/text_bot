from core import Bot


def show() -> None:
    print("=" * 60)
    print("Консольний ШІ-помічник")
    print("Інструкція: введіть ваше запитання та натисніть Enter.")
    print("Для завершення роботи введіть 'вихід' або 'exit'.")
    print("=" * 60)

    try:
        bot = Bot()
    except Exception as e:
        print(f"Критична помилка ініціалізації моделі [{type(e).__name__}]: {e}")
        return

    while True:
        try:
            user_input = input("\nВи: ").strip()

            match user_input.lower():
                case "":
                    continue
                case "вихід" | "exit":
                    print("Бот: Завершення роботи. До побачення!")
                    break
                case _:
                    print("Бот генерує відповідь...", end="\r")
                    response = bot.generate_response(user_input)
                    print(f"Бот: {response}")

        except KeyboardInterrupt:
            print("\nБот: Роботу перервано користувачем (Ctrl+C). До побачення!")
            break
        except Exception as e:
            print(f"\nПомилка під час генерації [{type(e).__name__}]: {e}")


if __name__ == "__main__":
    show()
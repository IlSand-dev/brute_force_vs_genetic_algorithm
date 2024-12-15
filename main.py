from datetime import timedelta

import pandas as pd

from additional_functions import segment_schedule
from brute_force import brute_force
from genetic import genetic
from config import min_time

if __name__ == "__main__":
    print("Это программа для составления расписания движения автобусов")
    print(
        """Есть два способа поиска лучшего расписания:
            1)Прямым перебором
            2)Используя генетический алгоритм""")
    search_type = 0
    search_function = 0
    kwargs = {}
    while True:
        search_type = input("Введите способ поиска расписания(1 или 2): ").strip()
        if search_type == "1":
            search_function = brute_force
            break
        elif search_type == "2":
            search_function = genetic
            print(
                """Хотите ли вы настроить алгоритм или хотите воспользоваться значениями по умолчанию?
                    1)По умолчанию(количество попыток = 10 000 количество расписаний = 1 000, шанс на рождение = 0,9, шанс мутации = 1e-09)
                    2)Настроить""")
            while True:
                setting_type = input("Введите способ задачи значений(1 или 2): ")
                if setting_type == "1":
                    break
                elif setting_type == "2":
                    while True:
                        attempts = input(
                            "Введите количество попыток(оставьте пустым чтобы использовать значение по умолчанию): ").strip()
                        if attempts == "":
                            break
                        elif attempts.isdecimal():
                            kwargs["attempts_amount"] = int(attempts)
                            break
                        else:
                            print("Количество попыток должно быть целым числом состоящим только из десятичных цифр")
                    while True:
                        schedules = input(
                            "Введите количество расписаний(оставьте пустым чтобы использовать значение по умолчанию): ").strip()
                        if schedules == "":
                            break
                        elif schedules.isdecimal():
                            kwargs["schedules_amount"] = int(schedules)
                            break
                        else:
                            print("Количество расписаний должно быть целым числом состоящим только из десятичных цифр")
                    while True:
                        birth_rate = input(
                            "Введите шанс рождения(оставьте пустым чтобы использовать значение по умолчанию): ").strip()
                        if birth_rate == "":
                            break
                        else:
                            try:
                                birth_rate = float(birth_rate)
                                if 0 < birth_rate < 1:
                                    kwargs["birth_rate"] = birth_rate
                                    break
                                else:
                                    print("Шанс рождения должен быть в интервале (0, 1)")
                            except ValueError:
                                print("Шанс рождения должен быть числом с плавающей точкой")
                    while True:
                        mutation_rate = input(
                            "Введите шанс мутации(оставьте пустым чтобы использовать значение по умолчанию): ").strip()
                        if mutation_rate == "":
                            break
                        else:
                            try:
                                mutation_rate = float(mutation_rate)
                                if 0 < mutation_rate < 1:
                                    kwargs["mutation_rate"] = mutation_rate
                                    break
                                else:
                                    print("Шанс мутации должен быть в интервале (0, 1)")
                            except ValueError:
                                print("Шанс мутации должен быть числом с плавающей точкой")
                    break
                else:
                    print("Введите 1 или 2")
            break
        else:
            print("Введите 1 или 2")
    type1 = 0
    type2 = 0
    while True:
        type1 = input("Введите число водителей с 8-ми часовой сменой: ")
        if type1.isdecimal():
            type1 = int(type1)
            if type1 >= 0:
                break
            else:
                print("Число водителей не может быть отрицательным")
        else:
            print("Число водителей должно быть числом")
    while True:
        type2 = input("Введите число водителей с 12-ти часовой сменой: ")
        if type2.isdecimal():
            type2 = int(type2)
            if type1 >= 0:
                break
            else:
                print("Число водителей не может быть отрицательным: ")
        else:
            print("Число водителей должно быть целым числом")

    best_schedule = search_function(type1, type2, **kwargs)
    schedules = []
    for i in range(len(best_schedule)):
        driver = best_schedule[i]
        segments = segment_schedule(driver)
        cur_time = 0
        start = 0
        day = 1
        schedules.append([f"Водитель {i+1}"] + [""] * 7)
        for j in range(len(segments)):
            segment = segments[j]
            cur_time += segment * min_time
            if cur_time // (24 * 60) > 0:
                day += 1
                cur_time %= 24 * 60
            if j % 2 == 0:
                start = cur_time
            else:
                if cur_time < start:
                    schedules[-1][day - 1] += f"{str(timedelta(minutes=int(start)))}-00:00 "
                    schedules[-1][day] += f"00:00-{str(timedelta(minutes=int(cur_time)))} "
                else:
                    schedules[-1][day] += f"{str(timedelta(minutes=int(start)))}-{str(timedelta(minutes=int(cur_time)))} "
        df = pd.DataFrame(schedules)
        df.columns = ["Номер водителя", "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота",
                      "Воскресенье"]
        df.to_csv("best_schedule.csv", index=False)

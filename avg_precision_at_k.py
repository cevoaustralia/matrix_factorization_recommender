from termcolor import colored


def average_precision_at_k(y_actual, y_pred, k_max=0):
    """
    Average Precision at k calculation
    """
    # Check if all elements in lists are unique
    if len(set(y_actual)) != len(y_actual):
        raise ValueError("Values in y_true are not unique")

    if len(set(y_pred)) != len(y_pred):
        raise ValueError("Values in y_pred are not unique")

    if k_max != 0:
        y_pred = y_pred[:k_max]

    num_relevant_items = 0
    running_sum = 0
    precision_at_k = []

    for i, yp_item in enumerate(y_pred):
        k = i + 1  # rank is 1-indexed

        if yp_item in y_actual:  # check if this item is in actual list
            num_relevant_items += 1
            running_sum += num_relevant_items / k
            precision_at_k.append(round((num_relevant_items / k), 2))
            print(
                f"Precision@{k}: {num_relevant_items}/{k} = {round((num_relevant_items / k), 2)}"
            )

    return (
        0 if num_relevant_items == 0 else round(running_sum / num_relevant_items, 5),
        precision_at_k,
        num_relevant_items,
    )


y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [1, 2, 7, 8, 9, 10]
print(colored("case 1: two correct predictions at the start of the list", "cyan"))
print(f"actual:     {y_actual}")
print(f"prediction: [{colored('1, 2', 'green')}, 7, 8, 9, 10]")
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)

y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 1, 3, 4, 8, 9]
print(
    colored(
        "case 2: two correct predictions plus relevant item in wrong position (before)",
        "cyan",
    )
)
print(f"actual:     {y_actual}")
print(
    f"prediction: [7, {colored('1', 'yellow')}, {colored('3', 'green')}, {colored('4', 'green')}, 8, 9]"
)
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]} + {avg_p_at_k[1][2]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)

y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 2, 8, 4, 9, 10]
print(colored("case 3: predictions for ranks 2 and 4 are correct", "cyan"))
print(f"actual:     {y_actual}")
print(f"prediction: [7, {colored('2', 'green')}, 8, {colored('4', 'green')}, 9, 10]")
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)

y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 8, 3, 4, 1, 9]
print(
    colored(
        "case 4: two correct predictions plus relevant item in wrong position (after)",
        "cyan",
    )
)
print(f"actual:     {y_actual}")
print(
    f"prediction: [7, 8, {colored('3', 'green')}, {colored('4', 'green')}, {colored('1', 'yellow')}, 9]"
)
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]} + {avg_p_at_k[1][2]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)

y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 8, 3, 4, 9, 10]
print(colored("case 5: two correct predictions in the middle of the list", "cyan"))
print(f"actual:     {y_actual}")
print(f"prediction: [7, 8, {colored('3', 'green')}, {colored('4', 'green')}, 9, 10]")
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)

y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 8, 9, 0, 5, 6]
print(colored("case 6: two correct predictions at the end of the list", "cyan"))
print(f"actual:     {y_actual}")
print(f"prediction: [7, 8, 9, 0, {colored('5', 'green')}, {colored('6', 'green')}]")
avg_p_at_k = average_precision_at_k(y_actual, y_pred)
print(
    f"Avg Precision@6: ({avg_p_at_k[1][0]} + {avg_p_at_k[1][1]})/{avg_p_at_k[2]} = {colored(avg_p_at_k[0], 'cyan')}\n"
)


y_actual = [1, 2, 3, 4, 5, 6]
y_pred = [7, 8, 9, 0, 10, 11]
print(colored("case 7: none of the predictions are correct", "cyan"))
print(f"actual:     {y_actual}")
print(f"prediction: {y_pred}")
print(
    f"Avg Precision@6: {colored(average_precision_at_k(y_actual, y_pred)[0], 'cyan')}\n"
)

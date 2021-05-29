from prettytable import PrettyTable


def show_recommendation(df, chosen_movie_index, movie_items_index):
    chosen_movie = df.iloc[chosen_movie_index]
    chosen_movie_description = smart_truncate(chosen_movie["description"])
    print(f"Given a movie: {chosen_movie['title']} - {chosen_movie_description}")
    table = PrettyTable()
    table.field_names = ["Films"]
    for index in movie_items_index:
        movie = df.iloc[index]
        movie_row = f"{movie['title']} - {smart_truncate(movie['description'])}"
        table.add_row([movie_row])
    print(table)


def smart_truncate(content, length=30, suffix='...'):
    if len(content) <= length:
        return content
    else:
        return ' '.join(content[:length+1].split(' ')[0:-1]) + suffix


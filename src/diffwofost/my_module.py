"""Documentation about the diffwofost module."""


# FIXME: put actual code here
def hello(name: str) -> str:
    """Say hello.

    Function docstring using Google docstring style.

    Args:
        name (str): Name to say hello to

    Returns:
        str: Hello message

    Raises:
        ValueError: If `name` is equal to `nobody`

    Example:
        This function can be called with `Jane Smith` as argument using

        >>> from diffwofost.my_module import hello
        >>> hello('Jane Smith')
        'Hello Jane Smith!'

    """
    if name == "nobody":
        msg = "Can not say hello to nobody"
        raise ValueError(msg)
    return f"Hello {name}!"

def calculate(expression: str):
    """Evaluate a mathematical expression
    :param expression: The mathematical expression to evaluate
    """
    try:
        result = eval(expression)
        return {"result": result}
    except:
        return {"error": "Invalid expression"}

def get_weather(location: str):
    """get the weather of a location
    :param location: where to get weather.
    """
    return {"temperature": "30C"}

def _run_python(code):
    allowed_globals = { '__builtins__': None, '_': None }
    allowed_locals = {}

    code = code.splitlines()
    code[-1] = f"_ = {code[-1]}"
    code = '\n'.join(code)

    try:
        exec(code, allowed_globals, allowed_locals)
    except Exception as e:
        return None

    return {'result': allowed_locals.get('_', None)}

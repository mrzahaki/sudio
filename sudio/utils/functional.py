import inspect
from functools import wraps
from typing import get_type_hints, Any, List, Dict, Union, Optional, _GenericAlias, get_origin, get_args, TypeVar, Generic
from types import UnionType

def get_function_param(func):
    return [param for param in inspect.signature(func).parameters.values()]

def get_param_names(func):
    return [param.name for param in inspect.signature(func).parameters.values()]



def check_generic_type(value, expected_type):
    if expected_type is Any:
        return True
    
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # print('origin:', origin, origin in (UnionType, Union))
    # print('args:', args)
    # print('value:', value)
    # print('\n')

    if value is None:
        if origin in (UnionType, Union) and type(None) in args:
            return True
        elif origin is None and type(None) == expected_type:
            return True
        return False
    
    if origin in (UnionType, Union):
        return any(check_generic_type(value, arg) for arg in args)
    if origin is None:
        return isinstance(value, expected_type)
    if not isinstance(value, origin):
        return False
    if origin == list:
        return all(check_generic_type(item, args[0]) for item in value)
    elif origin == dict:
        return all(check_generic_type(k, args[0]) and check_generic_type(v, args[1]) 
                  for k, v in value.items())
    
    return True

def type_check(check_args:bool=True, check_return:bool=True):

    def check(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            type_hints = get_type_hints(func)

            if check_args:
                sig = inspect.signature(func)
                params = list(sig.parameters.items())
                is_method = inspect.ismethod(func) if hasattr(func, '__self__') else False

                if not is_method and params and params[0][0] in ('self', 'cls'):
                    args_to_check = args[1:]
                    param_items = params[1:]
                else:
                    args_to_check = args
                    param_items = params

                # print('name: ', func.__name__)
                # print('args_to_check:', args_to_check)
                # print('param_items:', param_items)
                # print('\n')

                for arg, (param_name, _) in zip(args_to_check, param_items):
                    expected_type = type_hints.get(param_name, Any)
                    if not check_generic_type(arg, expected_type):
                        type_name = str(expected_type)
                        if get_origin(expected_type) in (UnionType, Union):
                            type_name = f"Union[{', '.join(str(t) for t in get_args(expected_type))}]"
                        raise TypeError(
                            f"Argument '{param_name}' in {func.__qualname__} must be of type {type_name}, "
                            f"got {type(arg).__name__}"
                        )
            
                    

            for name, arg in kwargs.items():
                if name in type_hints:
                    expected_type = type_hints[name]
                    if not check_generic_type(arg, expected_type):
                        type_name = str(expected_type)
                        if get_origin(expected_type) in (UnionType, Union):
                            type_name = f"Union[{', '.join(str(t) for t in get_args(expected_type))}]"
                        raise TypeError(
                            f"Argument '{name}' in {func.__qualname__} must be of type {type_name}, "
                            f"got {type(arg).__name__}"
                        )
            
            result = func(*args, **kwargs)

            if check_return and 'return' in type_hints:
                expected_return_type = type_hints['return']
                if not check_generic_type(result, expected_return_type):
                    type_name = str(expected_return_type)
                    if get_origin(expected_return_type) in (UnionType, Union):
                        type_name = f"Union[{', '.join(str(t) for t in get_args(expected_return_type))}]"
                    raise TypeError(
                        f"Return value of '{func.__name__}' must be of type {type_name}, "
                        f"got {type(result).__name__}"
                    )
            
            return result
        
        return wrapper
    return check


def get_super_classes(obj):
    """
    Returns all superclasses of an object in inheritance order.
    
    Args:
        obj: Any Python object or class
    
    Returns:
        list: List of all superclasses in method resolution order
    """
    if not isinstance(obj, type):
        obj = obj.__class__
        
    return obj.__mro__[1:-1] 


def get_class_hierarchy(obj):
    """
    Returns all superclasses + itself of an object in inheritance order.
    
    Args:
        obj: Any Python object or class
    
    Returns:
        list: List of all superclasses + main class itself in method resolution order
    """
    if not isinstance(obj, type):
        obj = obj.__class__
        
    return obj.__mro__[:-1] 


def _get_inherited_attributes(classes_list, property_name):
    """
    Get values of an inherited attribute from a list of classes in the inheritance chain.
    
    Args:
        classes_list (list): List of classes to check
        property_name (str): Name of the attribute to get
        
    Returns:
        list: List of attribute values from each class
    
    Raises:
        AttributeError: If attribute doesn't exist in any class
    """
    values = []
    
    for cls in classes_list:
        try:
            value = getattr(cls, property_name)
            values.append(value)
        except AttributeError:
            print(f"Warning: {cls.__name__} does not have property '{property_name}'")
            continue
            
    return values


def get_inherited_attributes(cls:object|list|tuple, property:object|list|tuple):
    """
    Gets attributes inherited from parent classes for specified classes and properties.

    This function retrieves inherited attributes by traversing the class hierarchy
    for given classes and properties. It handles both single classes/properties and
    lists/tuples of classes/properties.

    Args:
        cls (object|list|tuple, optional): Class(es) to check for inherited attributes.
            Can be a single class object or list/tuple of classes. Defaults to None.
        property (object|list|tuple, optional): Property/properties to search for in class hierarchy.
            Can be a single property object or list/tuple of properties. Defaults to None.

    Returns:
        list: List of inherited attributes found in the class hierarchy for given properties.
            Returns empty list if cls or property is None.
    """
    inherited_attributes = set()
    if isinstance(cls, (list, tuple)):
        for idx, c in enumerate(cls):
            if isinstance(property, (list, tuple)):
                inherited_attributes.update(_get_inherited_attributes(get_class_hierarchy(c), property[idx]))
            else:
                inherited_attributes.update(_get_inherited_attributes(get_class_hierarchy(c), property))
    elif isinstance(property, (list, tuple)):
        for idx, p in enumerate(property):
            if isinstance(cls, (list, tuple)):
                inherited_attributes.update(_get_inherited_attributes(get_class_hierarchy(cls[idx]), p))
            else:
                inherited_attributes.update(_get_inherited_attributes(get_class_hierarchy(cls), p))
    else:
        inherited_attributes.update(_get_inherited_attributes(get_class_hierarchy(cls), property))

    return tuple(inherited_attributes)


def exclude_kwargs(*args, cls:object|list|tuple=None, property:object|list|tuple=None, **kwargs):
    """
    Exclude specific keyword arguments from a dictionary of keyword arguments.
    This function allows you to exclude certain keyword arguments based on 
    the properties of a given class and its related classes.
    
    Parameters:
    *args: 
        Variable length argument list of properties to exclude.
    cls: type, list, or tuple, optional
        The class or list/tuple of classes whose related classes' properties are to be considered.
    property: str, list, or tuple, optional
        The property name or list/tuple of property names to look for in the related classes.
    **kwargs: 
        Arbitrary keyword arguments.
    
    Returns:
    dict:
        A dictionary containing the keyword arguments excluding the specified properties.
    """
    if cls is not None or property is not None:
        args += get_inherited_attributes(cls, property)

    all_param_names = set()
    for prop in args:
        all_param_names.update(get_param_names(prop))
    return {k: v for k, v in kwargs.items() if k not in all_param_names}


def filter_kwargs(*args, cls:object|list|tuple=None, property:object|list|tuple=None, **kwargs):
    """
    Filters keyword arguments based on the provided class and property attributes.
    Args:
        *args: Additional arguments that can be used to specify properties.
        cls (object | list | tuple, optional): A class or a list/tuple of classes to get inherited attributes from.
        property (object | list | tuple, optional): A property or a list/tuple of properties to get parameter names from.
        **kwargs: Keyword arguments to be filtered.
    Returns:
        dict: A dictionary containing only the keyword arguments that match the parameter names of the specified properties or class attributes.
    """

    if cls is not None or property is not None:
        args += get_inherited_attributes(cls, property)

    all_param_names = set()
    for prop in args:
        all_param_names.update(get_param_names(prop))

    return {k: v for k, v in kwargs.items() if k in all_param_names}


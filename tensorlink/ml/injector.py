import types
from typing import List, Dict, Set
import textwrap
import ast
import inspect

import torch.nn


class VariableUsageAnalyzer(ast.NodeVisitor):
    """Analyzes variable reads and writes within a code block"""

    def __init__(self):
        self.variables_read = set()
        self.variables_written = set()
        self.first_access = {}  # Track whether first access is read or write
        self.access_order = []  # Track order of all accesses

    def visit_Name(self, node):
        var_name = node.id

        if isinstance(node.ctx, ast.Load):
            self.variables_read.add(var_name)
            self.access_order.append(('read', var_name))
            if var_name not in self.first_access:
                self.first_access[var_name] = 'read'

        elif isinstance(node.ctx, ast.Store):
            self.variables_written.add(var_name)
            self.access_order.append(('write', var_name))
            if var_name not in self.first_access:
                self.first_access[var_name] = 'write'

        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        """Handle walrus operator assignments like: if (x := value)"""
        # The target of a walrus operator is always a write
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.variables_written.add(var_name)
            self.access_order.append(('write', var_name))
            if var_name not in self.first_access:
                self.first_access[var_name] = 'write'

        # Visit the value expression (might read other variables)
        self.visit(node.value)

    def visit_Attribute(self, node):
        """Track attribute access like self.x or decoder_layer.attention_type"""
        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{node.attr}"

            if isinstance(node.ctx, ast.Load):
                self.variables_read.add(full_name)
                self.access_order.append(('read', full_name))
                if full_name not in self.first_access:
                    self.first_access[full_name] = 'read'

            elif isinstance(node.ctx, ast.Store):
                self.variables_written.add(full_name)
                self.access_order.append(('write', full_name))
                if full_name not in self.first_access:
                    self.first_access[full_name] = 'write'

        self.generic_visit(node)


class LoopFinder(ast.NodeVisitor):
    """Find the layer iteration loop"""

    def __init__(self):
        self.loop_node = None

    def visit_For(self, node):
        """Find loop that iterates over layers"""
        # Look for patterns like: for ... in self.layers
        if isinstance(node.iter, ast.Attribute):
            if node.iter.attr in ['layers', 'h', 'blocks', 'decoder_layers']:
                self.loop_node = node
                return

        # Look for: for ... in self.layers[:...]
        if isinstance(node.iter, ast.Subscript):
            if isinstance(node.iter.value, ast.Attribute):
                if node.iter.value.attr in ['layers', 'h', 'blocks', 'decoder_layers']:
                    self.loop_node = node
                    return

        self.generic_visit(node)


class LayerCallExtractor(ast.NodeVisitor):
    """Extract the layer call from loop body"""

    def __init__(self):
        self.layer_calls = []

    def visit_Assign(self, node):
        """Find assignments that call the layer"""
        if isinstance(node.value, ast.Call):
            # Get the variable being assigned to (e.g., hidden_states)
            if isinstance(node.targets[0], ast.Name):
                assigned_to = node.targets[0].id
            else:
                assigned_to = ast.unparse(node.targets[0])

            # Extract call arguments
            call_info = self._extract_call_info(node.value)
            call_info['assigned_to'] = assigned_to

            self.layer_calls.append(call_info)

        self.generic_visit(node)

    def _extract_call_info(self, call_node: ast.Call) -> Dict:
        """Extract arguments from a function call"""
        # Get positional arguments
        positional_args = [ast.unparse(arg) for arg in call_node.args]

        # Get keyword arguments
        keyword_args = {}
        has_var_kwargs = False

        for keyword in call_node.keywords:
            if keyword.arg is None:  # **kwargs
                has_var_kwargs = True
            else:
                keyword_args[keyword.arg] = ast.unparse(keyword.value)

        return {
            'args': positional_args,
            'kwargs': keyword_args,
            'has_var_kwargs': has_var_kwargs,
        }


class FunctionArgExtractor(ast.NodeVisitor):
    """Extract function arguments"""

    def __init__(self):
        self.args = set()
        self.kwarg_name = None

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg != 'self':
                self.args.add(arg.arg)

            if node.args.kwarg:
                self.kwarg_name = node.args.kwarg.arg


class LayerGroupModule(torch.nn.Module):
    """
    Generic wrapper that executes the exact loop body for a subset of layers.
    Works for any model by reconstructing the original loop logic.
    """

    def __init__(
        self,
        layers: List[torch.nn.Module],
        input_vars: List[str],
        output_vars: List[str],
        loop_body_source: str,
        loop_iterator_name: str,
        debug: bool = True,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.loop_iterator_name = loop_iterator_name
        self.num_layers = len(layers)
        self.debug = debug

        # Generate the forward function from the loop body
        self.forward_func = self._generate_forward_from_loop(loop_body_source)

    def _generate_forward_from_loop(
        self, loop_body_source: str, debug: bool = False
    ) -> types.FunctionType:
        """
        Generate a forward function that executes the original loop body
        for each layer in self.layers.

        Args:
            loop_body_source: The source code of the original loop body
            debug: If True, add print statements for debugging
        """
        # Build function signature
        func_lines = [
            "def forward(self, **kwargs):",
            "    # Extract input variables",
        ]

        if debug:
            func_lines.append(
                "    print(f'[LayerGroupModule] Forward called with {len(kwargs)} kwargs')"
            )
            func_lines.append(
                "    print(f'[LayerGroupModule] Input kwargs keys: {list(kwargs.keys())}')"
            )

        # Extract each input variable from kwargs
        for var in self.input_vars:
            if var.endswith('_kwargs') or var == 'flash_attn_kwargs':
                func_lines.append(f"    {var} = kwargs.get('{var}', {{}})")
            else:
                func_lines.append(f"    {var} = kwargs.get('{var}')")

            if debug:
                func_lines.append(
                    f"    print(f'[LayerGroupModule] Extracted {var}: {{type({var})}}{{f\" shape={{list({var}.shape)}}\" if hasattr({var}, \"shape\") else \"\"}}')"
                )

        func_lines.append("")
        func_lines.append("    # Process through layers")
        if debug:
            func_lines.append(
                f"    print(f'[LayerGroupModule] Processing {{len(self.layers)}} layers')"
            )

        func_lines.append(
            f"    for layer_idx, {self.loop_iterator_name} in enumerate(self.layers):"
        )

        if debug:
            func_lines.append(
                f"        print(f'[LayerGroupModule] Layer {{layer_idx}}/{{len(self.layers)}}')"
            )
            func_lines.append(
                f"        print(f'[LayerGroupModule]   hidden_states before: {{hidden_states.shape if hasattr(hidden_states, \"shape\") else type(hidden_states)}} min={{hidden_states.min().item() if hasattr(hidden_states, \"min\") else \"N/A\"}} max={{hidden_states.max().item() if hasattr(hidden_states, \"max\") else \"N/A\"}} mean={{hidden_states.mean().item() if hasattr(hidden_states, \"mean\") else \"N/A\"}}')"
            )

        # Add the original loop body (indented appropriately)
        loop_lines = loop_body_source.strip().split('\n')
        for line in loop_lines:
            func_lines.append(f"        {line}")

        if debug:
            func_lines.append(
                f"        print(f'[LayerGroupModule]   hidden_states after: {{hidden_states.shape if hasattr(hidden_states, \"shape\") else type(hidden_states)}} min={{hidden_states.min().item() if hasattr(hidden_states, \"min\") else \"N/A\"}} max={{hidden_states.max().item() if hasattr(hidden_states, \"max\") else \"N/A\"}} mean={{hidden_states.mean().item() if hasattr(hidden_states, \"mean\") else \"N/A\"}}')"
            )
            func_lines.append(
                f"        if hasattr(hidden_states, 'isnan') and hidden_states.isnan().any():"
            )
            func_lines.append(
                f"            print(f'[LayerGroupModule]   WARNING: NaN detected in hidden_states after layer {{layer_idx}}')"
            )

        func_lines.append("")
        func_lines.append("    # Return outputs as a dictionary")
        if len(self.output_vars) == 0:
            if debug:
                func_lines.append(
                    "    print('[LayerGroupModule] Returning empty dict')"
                )
            func_lines.append(f"    return {{}}")
        else:
            # Build a dictionary with all output variables
            output_items = [f"'{var}': {var}" for var in sorted(self.output_vars)]
            if debug:
                func_lines.append(
                    f"    print(f'[LayerGroupModule] Returning {len(self.output_vars)} outputs')"
                )
                for var in sorted(self.output_vars):
                    func_lines.append(
                        f"    print(f'[LayerGroupModule] Output {var}: {{type({var})}}{{f\" shape={{list({var}.shape)}}\" if hasattr({var}, \"shape\") else \"\"}}')"
                    )
            func_lines.append(f"    return {{{', '.join(output_items)}}}")

        forward_source = '\n'.join(func_lines)

        if debug:
            print("=" * 80)
            print("GENERATED FORWARD SOURCE (LayerGroupModule)")
            print("=" * 80)
            print(forward_source)
            print("=" * 80)

        # Compile and return
        namespace = {'self': self, 'torch': torch}
        exec(forward_source, namespace)
        return namespace['forward']

    def forward(self, **kwargs):
        """
        Execute the generated forward function.
        This gets replaced by _generate_forward_from_loop.
        """
        return self.forward_func(self, **kwargs)


def _determine_loop_variables(
    var_analyzer: VariableUsageAnalyzer,
    loop_node: ast.For,
    function_args: Set[str],
    pre_loop_vars: Set[str],
    kwarg_name: str = None,
) -> Dict:
    """
    Determine which variables need to be passed to workers and returned.

    Only includes variables that:
    - Exist before the loop (simple names only)
    - Don't reference loop iterator or its attributes
    """
    # Get iterator variable (e.g., 'decoder_layer' in 'for decoder_layer in self.decoder_layers')
    if isinstance(loop_node.target, ast.Name):
        iterator_var = loop_node.target.id
    else:
        iterator_var = ast.unparse(loop_node.target)

    # Filter out iterator, self.*, and dotted names
    variables_read = set()
    for v in var_analyzer.variables_read:
        if v == iterator_var or v.startswith(f"{iterator_var}."):
            continue
        if v.startswith('self.'):
            continue
        if '.' not in v:
            variables_read.add(v)

    variables_written = set()
    for v in var_analyzer.variables_written:
        if v == iterator_var or v.startswith(f"{iterator_var}."):
            continue
        if v.startswith('self.'):
            continue
        if '.' not in v:
            variables_written.add(v)

    # A variable must exist before the loop if:
    # 1. It's first accessed via READ (not write), OR
    # 2. It's a function argument, OR
    # 3. It was created before the loop (NEW)
    read_before_write = set()
    for var in variables_read:
        if var in var_analyzer.first_access:
            if var_analyzer.first_access[var] == 'read':
                read_before_write.add(var)

    # Add variables that were created before the loop and are read in the loop
    read_before_write.update(variables_read & pre_loop_vars)

    # Add kwargs parameter if it exists and is used
    if kwarg_name and kwarg_name in variables_read:
        read_before_write.add(kwarg_name)
        function_args.add(kwarg_name)

    pre_loop_written_in_loop = pre_loop_vars & variables_written

    # Passthrough: variables both read and written (state that accumulates)
    # OR variables created before loop and modified in loop
    passthrough_vars = (
        (read_before_write & variables_written)
        | (function_args & variables_written)
        | pre_loop_written_in_loop
    )

    # Input-only: read but never written
    input_only_vars = (read_before_write - variables_written) | (
        function_args - variables_written
    )

    # Output-only: created in loop (written first) and never read
    output_only_vars = set()
    for var in variables_written:
        if var in var_analyzer.first_access:
            if var_analyzer.first_access[var] == 'write' and var not in variables_read:
                output_only_vars.add(var)

    # Filter inputs to only include function arguments or passthrough vars or pre-loop vars
    valid_input_only = input_only_vars & (function_args | pre_loop_vars)

    return {
        'input_vars': valid_input_only,
        'passthrough_vars': passthrough_vars,
        'output_vars': output_only_vars,
        'all_inputs': valid_input_only | passthrough_vars,
        'all_outputs': output_only_vars | passthrough_vars,
    }


def _extract_layer_call(loop_node: ast.For) -> Dict:
    """Extract the exact layer call signature from the loop body"""
    call_extractor = LayerCallExtractor()
    for stmt in loop_node.body:
        call_extractor.visit(stmt)

    if not call_extractor.layer_calls:
        raise ValueError("Could not find layer call in loop body")

    return call_extractor.layer_calls[0]


def _generate_new_forward_source(
    original_source: str,
    loop_node: ast.For,
    layer_call_info: Dict,
    loop_vars: Dict,
    offloaded_modules: List,
) -> str:
    """Generate new forward source with loop replaced by worker calls"""
    lines = original_source.split('\n')

    # Extract original function node to determine signature span
    module_ast = ast.parse(original_source)
    func_node = next(
        (
            n
            for n in module_ast.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        ),
        None,
    )

    if func_node is None:
        raise RuntimeError("Could not locate original forward()")

    # Find where the function signature starts and ends in the original source
    func_start_line = func_node.lineno - 1  # 0-indexed
    func_body_start_line = func_node.body[0].lineno - 1  # First statement in body

    # Manually construct clean signature without type hints
    clean_sig = _manually_construct_signature(func_node)

    # Extract sections
    before_func = lines[:func_start_line]

    # Skip decorators in before_func
    clean_before_func = []
    for line in before_func:
        if not line.strip().startswith('@'):
            clean_before_func.append(line)

    # Get the body (starting from first statement, skipping original signature)
    func_body = lines[func_body_start_line : loop_node.lineno - 1]

    after_loop = (
        lines[loop_node.end_lineno :] if loop_node.end_lineno < len(lines) else []
    )

    # Get indentation from the first body line
    if func_body:
        first_body_line = func_body[0]
        indent = len(first_body_line) - len(first_body_line.lstrip())
    else:
        # Fallback to loop indentation
        loop_line = lines[loop_node.lineno - 1]
        indent = len(loop_line) - len(loop_line.lstrip())

    indent_str = ' ' * indent

    # Generate worker calls
    worker_calls = _generate_worker_calls(
        layer_call_info, loop_vars, indent_str, offloaded_modules
    )

    # Combine
    new_lines = []
    new_lines.extend(clean_before_func)

    # Add the clean signature (with proper indentation if needed)
    func_indent = ' ' * (
        len(lines[func_start_line]) - len(lines[func_start_line].lstrip())
    )
    new_lines.append(func_indent + clean_sig)

    # Add the function body before the loop
    new_lines.extend(func_body)

    # Add worker calls
    new_lines.append(f"{indent_str}# Offloaded layer groups")
    new_lines.extend(worker_calls)

    # Add everything after the loop
    new_lines.extend(after_loop)

    return '\n'.join(new_lines)


def _manually_construct_signature(func_node: ast.FunctionDef) -> str:
    """Manually construct a clean function signature without type hints"""
    args_parts = []

    # Regular args with defaults
    num_defaults = len(func_node.args.defaults)
    num_args = len(func_node.args.args)

    for i, arg in enumerate(func_node.args.args):
        arg_str = arg.arg
        # Defaults align to the end of args list
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            default = func_node.args.defaults[default_idx]
            arg_str += f"={ast.unparse(default)}"
        args_parts.append(arg_str)

    # *args
    if func_node.args.vararg:
        args_parts.append(f"*{func_node.args.vararg.arg}")

    # Keyword-only args
    for arg, default in zip(func_node.args.kwonlyargs, func_node.args.kw_defaults):
        arg_str = arg.arg
        if default is not None:
            arg_str += f"={ast.unparse(default)}"
        args_parts.append(arg_str)

    # **kwargs
    if func_node.args.kwarg:
        args_parts.append(f"**{func_node.args.kwarg.arg}")

    return f"def {func_node.name}({', '.join(args_parts)}):"


def _generate_worker_calls(
    layer_call_info: Dict, loop_vars: Dict, indent: str, offloaded_modules: List
) -> List[str]:
    """Generate worker calls with proper variable passing and unpacking"""
    calls = []

    all_inputs = sorted(loop_vars['all_inputs'])
    all_outputs = sorted(loop_vars['all_outputs'])

    for idx, offloaded_module in enumerate(offloaded_modules):
        layer_range = getattr(offloaded_module, 'layer_range', 'unknown')

        # Comment
        calls.append(f"{indent}# Worker {idx}: layers {layer_range}")

        # Build the call
        call_str = f"{indent}_worker_output = self.offloaded_modules[{idx}]("

        # Add all input variables as keyword arguments
        arg_parts = []
        for var in all_inputs:
            arg_parts.append(f"{indent}    {var}={var}")

        # Add keyword arguments from original call that aren't already in inputs
        for kw_arg, kw_value in layer_call_info['kwargs'].items():
            if kw_arg not in all_inputs:
                arg_parts.append(f"{indent}    {kw_arg}={kw_value}")

        # Add **kwargs if present in original
        if layer_call_info['has_var_kwargs']:
            arg_parts.append(f"{indent}    **flash_attn_kwargs")

        if arg_parts:
            call_str += "\n" + ",\n".join(arg_parts) + f"\n{indent}"

        call_str += ")"
        calls.append(call_str)

        # Unpack the dictionary output
        if all_outputs:
            for var in all_outputs:
                calls.append(f"{indent}{var} = _worker_output.get('{var}', {var})")

        calls.append("")

    return calls


def _analyze_forward(fn):
    """
    Extract source, parse AST, extract args, and locate loop node.
    """
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    arg_extractor = FunctionArgExtractor()
    arg_extractor.visit(tree)

    loop_finder = LoopFinder()
    loop_finder.visit(tree)

    return source, tree, arg_extractor, loop_finder


def generate_new_forward_method(
    parent_module, base_module, offloaded_modules: List
) -> types.FunctionType:
    """
    Generate a new forward method with loop replaced by worker calls.

    Args:
        parent_module: The module whose forward pass contains the loop
        base_module: Base model containing the module
        offloaded_modules: List of OffloadedModule instances to call sequentially

    Returns:
        New forward function (unbound)
    """
    original_forward = parent_module.forward

    # First attempt: parent forward
    source, tree, arg_extractor, loop_finder = _analyze_forward(original_forward)

    # Fallback: base forward
    if not loop_finder.loop_node:
        try:
            original_forward = base_module.forward
            source, tree, arg_extractor, loop_finder = _analyze_forward(
                original_forward
            )
        except Exception:
            raise ValueError("No suitable loop found in forward pass")

    if not loop_finder.loop_node:
        raise ValueError("No suitable loop found in forward pass")

    # Analyze variable usage in loop
    loop_analyzer = VariableUsageAnalyzer()
    for stmt in loop_finder.loop_node.body:
        loop_analyzer.visit(stmt)

    # Analyze variables created BEFORE the loop
    func_node = tree.body[0]
    pre_loop_analyzer = VariableUsageAnalyzer()
    for stmt in func_node.body:
        if stmt == loop_finder.loop_node:
            break
        pre_loop_analyzer.visit(stmt)

    pre_loop_vars = pre_loop_analyzer.variables_written

    # Extract the layer call to preserve kwargs
    layer_call_info = _extract_layer_call(loop_finder.loop_node)

    # Determine input and output variables
    loop_vars = _determine_loop_variables(
        loop_analyzer,
        loop_finder.loop_node,
        arg_extractor.args,
        pre_loop_vars,
        arg_extractor.kwarg_name,
    )

    # Generate new forward code
    new_forward_code = _generate_new_forward_source(
        source,
        loop_finder.loop_node,
        layer_call_info,
        loop_vars,
        offloaded_modules,
    )

    # Prepare namespace
    namespace = _get_model_module_globals(parent_module, original_forward)

    try:
        exec(new_forward_code, namespace)
        return namespace["forward"]
    except Exception as e:
        print("=" * 80)
        print("ERROR COMPILING NEW FORWARD")
        print("=" * 80)
        print(new_forward_code)
        print("=" * 80)
        raise RuntimeError(f"Failed to compile: {e}")


def get_loop_io_signature(parent_module) -> Dict:
    """
    Analyze the forward pass loop and return all information needed
    to create model-agnostic layer group wrappers.

    Returns a dictionary with:
    - 'input_vars': Variables read but not written (function args only)
    - 'passthrough_vars': Variables both read and written (accumulating state)
    - 'output_vars': Variables written but not read
    - 'all_inputs': All variables that should be passed in
    - 'all_outputs': All variables that should be returned
    - 'layer_call_info': Information about the original layer call signature
    - 'loop_body_source': The exact source code of the loop body
    - 'loop_iterator_name': Name of the loop iterator variable
    """
    # Find the module that contains the loop
    module_with_loop, loop_node, module_path = find_loop_in_module_hierarchy(
        parent_module
    )

    # Get original forward source
    original_forward = module_with_loop.forward
    source = inspect.getsource(original_forward)
    source = textwrap.dedent(source)

    # Parse and analyze
    tree = ast.parse(source)

    # Extract function arguments
    arg_extractor = FunctionArgExtractor()
    arg_extractor.visit(tree)

    # Find the loop
    loop_finder = LoopFinder()
    loop_finder.visit(tree)

    if not loop_finder.loop_node:
        raise ValueError("No suitable loop found in forward pass")

    # Analyze variable usage in loop
    loop_analyzer = VariableUsageAnalyzer()
    for stmt in loop_finder.loop_node.body:
        loop_analyzer.visit(stmt)

    # Analyze variables created before the loop
    func_node = tree.body[0]  # The forward function
    pre_loop_analyzer = VariableUsageAnalyzer()
    for stmt in func_node.body:
        # Stop when we reach the loop
        if stmt == loop_finder.loop_node:
            break
        pre_loop_analyzer.visit(stmt)

    # Variables that exist before the loop are those written before it
    pre_loop_vars = pre_loop_analyzer.variables_written

    # Extract the layer call to preserve kwargs
    call_extractor = LayerCallExtractor()
    for stmt in loop_finder.loop_node.body:
        call_extractor.visit(stmt)

    layer_call_info = (
        call_extractor.layer_calls[0] if call_extractor.layer_calls else {}
    )

    # Determine input and output variables
    loop_vars = _determine_loop_variables(
        loop_analyzer,
        loop_finder.loop_node,
        arg_extractor.args,
        pre_loop_vars,
        arg_extractor.kwarg_name,
    )

    # Extract loop body source and iterator name
    loop_body_source = _extract_loop_body_source(loop_finder.loop_node, source)
    loop_iterator_name = _extract_loop_iterator_name(loop_finder.loop_node)

    # Add all extracted info
    result = {
        **loop_vars,
        'layer_call_info': layer_call_info,
        'loop_body_source': loop_body_source,
        'loop_iterator_name': loop_iterator_name,
        'module_with_loop': module_with_loop,
        'module_path': module_path,
    }

    return result


def _get_model_module_globals(parent_module, original_forward) -> dict:
    """
    Get the correct global namespace for the model's forward method.

    This handles cases where the forward method might be wrapped/decorated
    and its __globals__ points to the wrong module.
    """
    import sys

    # Attempt 1: Get from the parent module's class
    parent_class = type(parent_module)
    class_module_name = parent_class.__module__

    if class_module_name in sys.modules:
        class_module = sys.modules[class_module_name]
        return vars(class_module)

    # Attempt 2: Try to get from the forward method's __globals__
    if hasattr(original_forward, '__func__'):
        func_globals = original_forward.__func__.__globals__
    elif hasattr(original_forward, '__globals__'):
        func_globals = original_forward.__globals__
    else:
        func_globals = {}

    # Attempt 3: If the globals look wrong (like transformers.utils.generic),
    # try to import the actual model module
    if '__name__' in func_globals:
        globals_module = func_globals['__name__']
        if 'utils' in globals_module or 'generic' in globals_module:
            # This is a utility module, not the actual model module
            # Try to get the real module
            if class_module_name in sys.modules:
                return vars(sys.modules[class_module_name])

    return func_globals


def _extract_loop_body_source(loop_node: ast.For, original_source: str) -> str:
    """
    Extract the exact source code of the loop body.
    This preserves the original logic for model-agnostic execution.
    """
    lines = original_source.split('\n')

    # Get the first and last line of the loop body
    first_stmt = loop_node.body[0]
    last_stmt = loop_node.body[-1]

    start_line = first_stmt.lineno - 1  # 0-indexed
    end_line = last_stmt.end_lineno  # inclusive, 1-indexed, so no -1

    # Extract the body lines
    body_lines = lines[start_line:end_line]

    # Determine the base indentation (from first line)
    if body_lines:
        first_line = body_lines[0]
        base_indent = len(first_line) - len(first_line.lstrip())

        # Remove base indentation from all lines
        dedented_lines = []
        for line in body_lines:
            if line.strip():  # Non-empty line
                dedented_lines.append(line[base_indent:])
            else:  # Empty line
                dedented_lines.append('')

        return '\n'.join(dedented_lines)

    return ''


def _extract_loop_iterator_name(loop_node: ast.For) -> str:
    """
    Extract the name of the loop iterator variable.
    Example: for decoder_layer in self.layers: -> 'decoder_layer'
    """
    if isinstance(loop_node.target, ast.Name):
        return loop_node.target.id
    else:
        return ast.unparse(loop_node.target)


def find_loop_in_module_hierarchy(parent_module, max_depth=2):
    """
    Find the layer loop, checking submodules if necessary.

    Args:
        parent_module: The top-level module to start from
        max_depth: Maximum depth to search in module hierarchy

    Returns:
        tuple: (module_with_loop, loop_node, module_path)
            - module_with_loop: The module containing the loop
            - loop_node: The AST node of the loop
            - module_path: List of attribute names to reach the module (e.g., ['model'])
    """

    def try_find_loop(module, path=[]):
        """Recursively try to find the loop in module or its submodules"""
        # Get the forward method source
        try:
            forward_method = module.forward
            source = inspect.getsource(forward_method)
            source = textwrap.dedent(source)
        except (TypeError, OSError):
            return None

        # Parse and look for loop
        tree = ast.parse(source)
        loop_finder = LoopFinder()
        loop_finder.visit(tree)

        if loop_finder.loop_node:
            return (module, loop_finder.loop_node, path)

        # If no loop found and we haven't reached max depth, check for delegation
        if len(path) < max_depth:
            delegated_attr = find_delegated_module_call(tree)
            if delegated_attr and hasattr(module, delegated_attr):
                submodule = getattr(module, delegated_attr)
                if hasattr(submodule, 'forward'):
                    return try_find_loop(submodule, path + [delegated_attr])

        return None

    result = try_find_loop(parent_module)
    if result is None:
        raise ValueError("No suitable loop found in forward pass or submodules")

    return result


def find_delegated_module_call(tree):
    """
    Find if the forward method delegates to a submodule.

    Looks for patterns like:
        outputs = self.model(...)
        return self.encoder(...)

    Returns:
        str: The attribute name of the delegated module (e.g., 'model', 'encoder')
             or None if no delegation found
    """

    class DelegationFinder(ast.NodeVisitor):
        def __init__(self):
            self.delegated_attrs = []

        def visit_Call(self, node):
            # Look for calls like: self.module(...)
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == 'self':
                        # This is a call to self.something(...)
                        # Check if it looks like a module call (not a method)
                        # Common module attributes: model, encoder, decoder, transformer
                        attr = node.func.attr
                        if attr in [
                            'model',
                            'encoder',
                            'decoder',
                            'transformer',
                            'bert',
                            'gpt',
                            'llama',
                            'mistral',
                        ]:
                            self.delegated_attrs.append(attr)

            self.generic_visit(node)

    finder = DelegationFinder()
    finder.visit(tree)

    # Return the first delegation found (usually there's only one main one)
    return finder.delegated_attrs[0] if finder.delegated_attrs else None

## Contents

* **[Contents](CodingConventions.md#contents)**
* **[Introduction](CodingConventions.md#introduction)**
* **[C99](CodingConventions.md#c99)**
  * [Placement of braces](CodingConventions.md#placement-of-braces)
  * [Indentation](CodingConventions.md#indentation)
  * [Comments](CodingConventions.md#comments)
  * [Blank lines](CodingConventions.md#blank-lines)
  * [Condensing short code to single lines](CodingConventions.md#condensing-short-code-to-single-lines)
  * [Whitespace in function calls](CodingConventions.md#whitespace-in-function-calls)
  * [Whitespace in function definitions](CodingConventions.md#whitespace-in-function-definitions)
  * [Whitespace in expressions](CodingConventions.md#whitespace-in-expressions)
  * [Trailing whitespace](CodingConventions.md#trailing-whitespace)

## Introduction

This wiki describes the coding conventions used in BLIS. Please try to adhere to these conventions when submitting pull requests and/or (if you have permission) committing directly to the repository.

There is some support for these conventions for Emacs editing in the `.dir-locals.el` file, which will affect editing with CC mode in the blis directory.

## C99

Most of the code in BLIS is written in C, and specifically in ISO C99. This section describes the C coding standards used within BLIS.

### Placement of braces

Please either use braces to denote the indentation limits of scope, or to enclose multiple statements on a single line. But do not place the open brace on the same line as a conditional if the conditional will be more than one line.
```c
{
    // This is fine.
    if ( bli_obj_is_real( x ) )
    {
        foo = 1;
    }

    // This is also fine. (Ideal for short conditional bodies.)
    if ( bli_obj_is_real( x ) ) { foo = 1; return; }

    // This is bad. Please use one of the two forms above.
    if ( bli_obj_is_real( x ) ) {
        foo = 1;
    }

    // This is (much) worse. Please no.
    if ( bli_obj_is_real( x ) )
        {
        foo = 1;
        }
}
```

### Indentation

If at all possible, **please use tabs to denote changing levels of scope!** If you can't use tabs or doing so would be very inconvenient given your editor and setup, please set your indentation to use exactly four spaces per level of indentation. Below is what it would look like if you used tabs (with a tab width set to occupy four spaces), or four actual spaces per indentation level.
```c
bool bli_obj_is_real( obj_t* x )
{
    bool r_val;

    if ( bli_obj_is_real( x ) )
        r_val = TRUE;
    else
        r_val = FALSE;
}
```
Ideally, tabs should be used to indicate changes in levels of scope, but then spaces should be used for multi-line statements within the same scope. In the example below, I've marked the characters that should be spaces with `.` (with tabs used for the first level of indentation):

```c
bool bli_obj_is_complex( obj_t* x )
{
    bool r_val;

    if ( bli_obj_is_scomplex( x ) ||
    .....bli_obj_is_dcomplex( x ) ) r_val = TRUE;
    else............................r_val = FALSE;

    return r_val;
}
```

### Comments

Please use C++-style comments, and line-break your comments somewhere between character (column) 72 and 80.
```c
{
    // This is a comment. This comment can span multiple lines, but it should 
    // not extend beyond column 80. (For these purposes, you can count a tab 
    // as anywhere from one to four spaces.)
}
```
If you are inserting comments in a macro definition, in which case you must use C-style comments:
```c
#define bli_some_macro( x ) \
\
    /* This is a comment in a macro definition. It, too, should not spill
       beyond column 80. Please place the ending comment marker on the last
       line containing words, unless the comment marker would cause you to
       go beyond column 80, in which case you can place it on the next line
       aligned with the first comment marker. */
```

### Blank lines

Please use blank lines to separate lines of code from the next line of code. However, if adjacent lines of code are meaningfully related, please skip the blank line.
```c
{
    // Set the matrix datatype.
    bli_obj_set_dt( BLIS_DOUBLE, x );

    // Set the matrix dimensions.
    bli_obj_set_length( 10, x );
    bli_obj_set_width( 5, x );

    // Set the matrix structure.
    bli_obj_set_struc( BLIS_GENERAL, x );
    bli_obj_set_uplo( BLIS_DENSE, x );
}
```

### Condensing short code to single lines

Sometimes, to more efficiently display code on the screen, it's helpful to skip certain newlines, such as those in conditional statements. This is fine, just try to line things up in a way that is visually appealing.
```c
{
    bool  r_val;
    dim_t foo;

    // This is fine.
    if ( bli_obj_is_real( x ) ) r_val = TRUE;
    else                        r_val = FALSE;

    // This is okay. (Notice the spaces after '{' and before '}'.)
    // However, the next example is preferred over this style.
    if ( bli_obj_is_real( x ) ) { r_val = TRUE; foo = 1; }
    else                        { r_val = FALSE; foo = 0; }

    // Similar to above, but with some extra alignment. This is better
    // than above.
    if ( bli_obj_is_real( x ) ) { r_val = TRUE;  foo = 1; }
    else                        { r_val = FALSE; foo = 0; }
}
```

### Whitespace in function calls

For single-line function calls, **please avoid** a space between the last character in the function/macro name and the open parentheses. Also, please do not insert any spaces before commas that separate arguments to a function/macro invocation. But please **do** insert at least once space after each comma. (I say "at least one" because sometimes it looks nicer to align the commas with those of function calls on lines above or below the function call in question.) Also, please include one space between the opening parentheses and the first argument, and also between the last argument and closing parentheses
```c
{
    obj_t x;

    // Good.
    bli_obj_create( BLIS_DOUBLE, 3, 4, 0, 0, &x );
    bli_obj_set_length( 10, x );

    // Bad. Please avoid these.
    bli_obj_set_dt ( BLIS_FLOAT, x );
    bli_obj_set_dt( BLIS_FLOAT , x );
    bli_obj_set_dt(BLIS_FLOAT, x);
    bli_obj_set_dt(BLIS_FLOAT,x);

    // Good.
    bli_obj_set_dt( BLIS_FLOAT, x );
}
```
For multi-line function calls, please use the following template:
```c
{
    bli_dgemm
    (
      BLIS_NO_TRANSPOSE,
      BLIS_TRANSPOSE,
      m, n, k,
      &BLIS_ONE
      a, rs_a, cs_a,
      b, rs_b, cs_b,
      &BLIS_ZERO,
      c, rs_c, cs_c
    );
}
```
Notice that here, the parentheses are formatted similar to braces. However, notice that the arguments do not constitute a new level of "scope." Instead, you should use exactly two additional spaces. before each line of arguments.

### Whitespace in function definitions

When defining a function with few arguments, insert a single space after commas and types, and after the first parentheses and before the last parentheses:
```c
// Please write "short" function signatures like this.
void bli_obj_set_length( dim_t m, obj_t* a )
{
    // Body of function
}
```
As with single-line function calls, please do not place a space between the last character of the function name and the open parentheses to the argument list!
```c
// Please avoid this.
void bli_obj_set_length ( dim_t m, obj_t* a )
{
    // Body of function
}
```
When defining a function with many arguments, especially those that would not comfortably fit in a single 80-character line, you can split the type signature into multiple lines:
```c
// Please write "long" function signatures like this.
void bli_gemm
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
    // Body of function
}
```
If you are going to use this style of function definition, please indent the parentheses exactly five spaces (don't use tabs here). Then, indent the arguments with an additional two spaces. Thus, parentheses should be in column 6 (counting from 1) and argument types should begin in column 8. Also notice that the number of spaces after each argument's type specifier varies so that the argument names are aligned. If you insert qualifiers such as `restrict`, please right-justify them:
```c
// Please align 'restrict' keywords and variables, as appropriate.
void bli_gemm
     (
       obj_t*  restrict alpha,
       obj_t*  restrict a,
       obj_t*  restrict b,
       obj_t*  restrict beta,
       obj_t*  restrict c,
       cntx_t* restrict cntx
     )
{
    // Body of function
}
```

### Whitespace in expressions

Please insert whitespace into conditional expressions.
```c
{
    // Good.
    if ( m == 10 && n > 0 ) return;

    // Bad.
    if ( m==10 && n>0 ) return;

    // Worse!
    if (m==10&&n>0) return;

    // Okay, now you're just messing with me.
    if(m==10&&n>0)return;
}
```
Unlike with the parentheses that surround the argument list of a function call, there should be exactly one space after conditional keywords and the open parentheses for its associated conditional statement: `if (...)`, `else if (...)`, and `while (...)`.
```c
{
    // Good.
    if ( ... ) return 0;
    else if ( ... ) return 1;

    // Good.
    while ( ... )
    {
        // loop body.
    }

    // Good.
    do
    {
        // loop body.
    } while ( ... );
}
```
Sometimes, extra spaces for alignment are desired:
```c
{
    // This is okay.
    if ( m == 0 ) return 0;
    else if ( n == 0 ) return 1;

    // This is sometimes preferred because it allows your eyes to more easily
    // see the differences between the 'if' conditional expression and the
    // 'else if' conditional expression.
    if      ( m == 0 ) return 0;
    else if ( n == 0 ) return 1;
}
```

### Trailing whitespace

Please try to avoid inserting any trailing whitespace. This also means that "blank" lines should not contain any tabs or spaces.

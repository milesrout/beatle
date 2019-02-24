# beatle

## features

 * strong, static typing
 * type inference
 * pleasant syntax inspired by Python
 * opt-in laziness
 * opt-in mutability
 * macros and quasiquoting (((without s-expressions)))
 * all the things people expect in modern programming languages
   - first-class functions
   - lambda expressions
   - algebraic data types (A ⨯ B; A | B; etc.)
   - pattern matching
   - tail call optimisation
   - parametric polymorphism
   - persistent data structures
   - memory safety
   - generators
 * intuitive defaults inspired by Python
   - numbers are arbitrary-precision (floats and fixed-precision integers
     are opt-in).
   - if at all possible, naive/obvious operations like append-in-a-loop and
     slicing sequences should be fast.
 * dynamically-scoped 'contexts' that avoid the need to thread the same
   variables through virtually every single function in a programme.
 * multi-paradigm support
   - *functional*, with clever syntax to allow multi-line lambdas in a
     Python-like syntax. First-class functions go without saying in this day
     and age.
   - *imperative*, with mutable variables when you need them.
   - *reactive*, with good support for 'reactive' cells that make an
     'interactive notebook' experience first-class.
   - *concurrent*, with green threads communicating with channels.
   - *relational*, with built-in support for tables, SQL-like queries and other
     relational programming constructs.
   - *meta*, with language extension using macros.
   - *reliable*, with a clear distinction between recoverable 'resumable
     conditions' and unrecoverable 'panics' instead of error-prone exceptions
     or return codes.

## interplay between paradigms

Functional and reactive programming go hand in hand: reactive cells only really
makes sense when outputs depend entirely on inputs.

Relational and reactive programming work very well together. The most widely
used reactive environments are spreadsheets! Construct new reactive cells from
queries over reactive tables for the equivalent of SQL's 'CREATE VIEW'.

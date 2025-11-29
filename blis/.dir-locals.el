;; Emacs formatting for the BLIS layout requirements.

(
 ;; Recognize *.mk files as Makefile fragments
 (auto-mode-alist . (("\\.mk\\'" . makefile-mode)) )

 ;; Makefiles require tabs and are almost always width 8
 (makefile-mode . (
                   (indent-tabs-mode . t)
                   (tab-width . 8)
                   )
                )

 ;; C code formatting roughly according to docs/CodingConventions.md
 (c-mode . (
            (c-file-style . "bsd")
            (c-basic-offset . 4)
            (comment-start . "// ")
            (comment-end . "")
            (parens-require-spaces . nil)
            )
         )

 ;; Default formatting for all source files not overriden above
 (prog-mode . (
               (indent-tabs-mode . nil)
               (tab-width . 4)
               (require-final-newline . t)
               (eval add-hook `before-save-hook `delete-trailing-whitespace)
               )
            )
)

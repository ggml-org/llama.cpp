/*-------------------------------------------------------------------------
 * Independent Linker Script for GGML Bare Metal Kernels
 *-------------------------------------------------------------------------
 */

/* Define maximum number of harts (threads) - simplified for bare metal */
MAX_HARTS = 2112;

SECTIONS
{
  /* Start at the base address passed by -Wl,--defsym=BASE_ADDRESS=... */
  . = BASE_ADDRESS;

  /* Export entry point symbol for runtime compatibility */
  KERNEL_UMODE_ENTRY = BASE_ADDRESS;

  /* Initialization section - must come first */
  .text.init :
  {
     *(.text.init)
  }

  /* Align to cache line boundary */
  . = ALIGN(64);

  /* Main text section for code */
  .text : {
    *(.text)
    *(.text.*)
  }
  . = ALIGN(64);

  /* Data section with global pointer setup */
  .data :
  {
    _data_start = .;
    . = ALIGN(64);

    /* Small data section and global pointer */
    *(.sdata .sdata.*)
    PROVIDE( __global_pointer$ = . + 0x800 );

    /* Regular data */
    *(.data .data.*)
    . = ALIGN(64);
    _data_end = .;
  }
  . = ALIGN(64);

  /* BSS section for uninitialized data */
  .bss(NOLOAD) :
  {
    _bss_start = .;
    *(.sbss*);
    *(.bss*);
    . = ALIGN(64);
    _bss_end = .;
  }

  /* Thread Local Storage (TLS) sections */
  . = ALIGN(64);
  .tdata :
  {
     *(.tdata*)
     . = ALIGN(64);
  }
  __tdata_start = ADDR(.tdata);

  .tbss : {
      __tbss_start = .;
      *(.tbss*)
  }
  . = . + SIZEOF(.tbss);
  . = ALIGN(64);
  __tbss_end = .;

  /* TLS allocation area for all harts */
  .tls-alloc ALIGN(64) (NOLOAD) : {
     __tls_alloc_start = .;
     . = . + (ABSOLUTE(__tbss_end) - ABSOLUTE(__tdata_start)) * MAX_HARTS;
  }

  /* End of kernel image */
  _end = .;
}

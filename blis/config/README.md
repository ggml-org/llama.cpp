
For more information on sub-configurations and configuration families in BLIS,
please read the Configuration Guide, which can be viewed in markdown-rendered
form [from the BLIS wiki page](https://github.com/flame/blis/wiki/).

If you don't have time, or are impatient, take a look at the `config_registry`
file in the top-level directory of the BLIS distribution. It contains a
grammar-like mapping of configuration names, or families, to sub-configurations,
which may be other families. Keep in mind that the `/` notation:
```
<config>: <config>/<name>
```
means that the kernel set associated with `<name>` should be made available to
 the configuration `<config>` if `<config>` is targeted at configure-time.
(Some configurations borrow kernels from other configurations, and this is how
we specify that requirement.)


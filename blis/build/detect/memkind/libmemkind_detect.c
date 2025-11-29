#include <stdio.h>
#include <hbwmalloc.h>

int main( int argc, char **argv )
{
    void* p = hbw_malloc( 4096 );

	printf( "%s: hbw_malloc() returned %p\n", __FILE__, p );

    return 0;
}


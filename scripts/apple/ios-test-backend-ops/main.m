// iOS host for test-backend-ops: runs it with the launch arguments, keeps the screen on while it runs,
// then exits with its return code (run.sh streams the output with `devicectl ... --console`).
#import <UIKit/UIKit.h>

#include <stdio.h>
#include <stdlib.h>

int ggml_run_test_backend_ops(int argc, char ** argv);

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (strong, nonatomic) UIWindow * window;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)options {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController * vc = [UIViewController new];
    UILabel * label = [[UILabel alloc] initWithFrame:vc.view.bounds];
    label.text = @"test-backend-ops running";
    label.textAlignment = NSTextAlignmentCenter;
    vc.view.backgroundColor = UIColor.systemBackgroundColor;
    [vc.view addSubview:label];
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
    app.idleTimerDisabled = YES;

    NSArray<NSString *> * args = NSProcessInfo.processInfo.arguments;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        setvbuf(stdout, NULL, _IOLBF, 0);
        setvbuf(stderr, NULL, _IOLBF, 0);
        int argc = (int) args.count;
        char ** argv = calloc(argc + 1, sizeof(char *));
        for (int i = 0; i < argc; i++) {
            argv[i] = strdup(args[i].UTF8String);
        }
        const int rc = ggml_run_test_backend_ops(argc, argv);
        printf("test-backend-ops exited with code %d\n", rc);
        fflush(stdout);
        exit(rc);
    });
    return YES;
}
@end

int main(int argc, char * argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass(AppDelegate.class));
    }
}

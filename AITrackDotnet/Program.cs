using AITrackDotnet.HostedServices;
using AITrackDotnet.Infrastructure.Serilog;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureLogging((context, logging) =>
    {
        SerilogConfiguration.ConfigureSerilog(logging, context.Configuration);
    })
    .ConfigureServices(services =>
    {
        services.AddHostedService<MainLoopHostedService>();
    });

var app = host.Build();
await app.RunAsync();
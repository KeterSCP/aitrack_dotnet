using AITrackDotnet;
using AITrackDotnet.HostedServices;
using AITrackDotnet.Infrastructure.Serilog;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Primitives;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureLogging((context, logging) =>
    {
        SerilogConfiguration.ConfigureSerilog(logging, context.Configuration);
    })
    .ConfigureServices(services =>
    {
        services.AddHostedService<MainLoopHostedService>();
    })
    .ConfigureAppConfiguration(conf =>
    {
        conf.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
    });

var app = host.Build();

var configuration = app.Services.GetRequiredService<IConfiguration>();
AppSettings.Load(configuration);
AppSettings.WasReloaded = false;

// See https://github.com/dotnet/aspnetcore/issues/2542
using var debouncer = new Debouncer(TimeSpan.FromMilliseconds(300));

ChangeToken.OnChange(configuration.GetReloadToken, () =>
{
    // ReSharper disable once AccessToDisposedClosure
    debouncer.Debounce(() =>
    {
        configuration = app.Services.GetRequiredService<IConfiguration>();
        AppSettings.Load(configuration);
    });
});

await app.RunAsync();
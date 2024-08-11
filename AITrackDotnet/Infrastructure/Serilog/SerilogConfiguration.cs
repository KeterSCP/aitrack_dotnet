using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;
using Serilog.Exceptions;

namespace AITrackDotnet.Infrastructure.Serilog;

public static class SerilogConfiguration
{
    public static void ConfigureSerilog(ILoggingBuilder loggingBuilder, IConfiguration configuration)
    {
        loggingBuilder.ClearProviders();

        var loggerConfiguration = new LoggerConfiguration()
            .Enrich.FromLogContext()
            .Enrich.WithExceptionDetails();

        loggerConfiguration.WriteTo.Console(LogEventLevel.Debug);

        Log.Logger = loggerConfiguration.CreateLogger();
        loggingBuilder.AddSerilog();
    }
}